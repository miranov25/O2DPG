"""Recursive semantic description with provenance — PHASE_13_82_DF, stage 1.

WHAT THIS IS
    A read-only description of what dfdraw believes a request means, and where
    each effective value came from. It draws nothing and changes nothing.

WHY IT EXISTS
    The meaning of a request is currently spread across call arguments, the
    global configuration registry in `style.py`, per-function defaults, and
    values worked out from the data. Nobody can see the whole picture at once.
    That makes three things harder than they should be: systematic testing,
    debugging "why did dfdraw do that", and changing the code without changing
    behaviour by accident.

THE ONE RULE THAT SHAPES THIS MODULE
    The proposal forbids a second semantic owner: production code deciding one
    thing while an explanation independently decides another, free to drift
    apart. So this module does NOT reimplement any resolution. It provides one
    helper that resolves a value AND records where it came from, and the
    production path calls that same helper. There is one implementation and two
    callers.

    Concretely, `plots/profile.py` previously did:

        if bins is None:
            bins = get_style_value("hist.bins", 50)

    It now calls `resolve()` instead, which does exactly the same thing and
    additionally records the origin. The resolved value is unchanged; only the
    bookkeeping is new.

TWO KINDS OF ORIGIN INFORMATION
    For most fields the origin is purely diagnostic: `bins=100` behaves
    identically whether you passed it or the global registry supplied it.

    For a few fields the origin is part of the meaning. If you pass
    `marker="s"`, every group gets that marker; if you pass nothing, each group
    takes a different marker from a cycle. Same value, different behaviour.
    Those fields are marked `origin_sensitive=True` here, and the production
    code has always tracked them (see the `_ud_user_*` captures in
    `plots/profile.py`). This module records that fact rather than inventing it.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
    It does not guess. The global registry stores effective values but not
    their history, so after a named style followed by a custom patch it is
    often impossible to say which one supplied an unchanged value. In that case
    the source is reported as CURRENT_GLOBAL_CONFIGURATION with
    origin_detail=UNKNOWN, rather than as a plausible invention. An explanation
    that made something up would be worse than one that admits the limit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# --------------------------------------------------------------------------
# Where a value can come from
# --------------------------------------------------------------------------

#: The caller passed it explicitly.
CALL_ARGUMENT = "CALL_ARGUMENT"

#: The current global configuration registry supplied it. Rev2 vocabulary:
#: the SOURCE is always this when the registry answered; how precisely we can
#: name the contributor is carried separately in `origin_detail`.
CURRENT_GLOBAL_CONFIGURATION = "CURRENT_GLOBAL_CONFIGURATION"

#: No registry key exists for this field; the call site's own fallback applied.
BUILTIN_DEFAULT = "BUILTIN_DEFAULT"

#: Worked out from the data (for example an automatic axis range).
DATA_DERIVED = "DATA_DERIVED"

#: Came from outside dfdraw — an existing axes object, matplotlib global state.
EXTERNAL_RUNTIME_CONTEXT = "EXTERNAL_RUNTIME_CONTEXT"

ORIGINS = (
    CALL_ARGUMENT,
    CURRENT_GLOBAL_CONFIGURATION,
    BUILTIN_DEFAULT,
    DATA_DERIVED,
    EXTERNAL_RUNTIME_CONTEXT,
)


# --------------------------------------------------------------------------
# Contract / implementation status vocabulary — PHASE_13_82_DF v1.1
# --------------------------------------------------------------------------

CONTRACT_SUPPORTED = "SUPPORTED"
CONTRACT_NOT_APPLICABLE = "NOT_APPLICABLE"
CONTRACT_REFUSE_BY_DESIGN = "REFUSE_BY_DESIGN"
CONTRACT_UNRESOLVED = "UNRESOLVED"
CONTRACT_STATUSES = (
    CONTRACT_SUPPORTED,
    CONTRACT_NOT_APPLICABLE,
    CONTRACT_REFUSE_BY_DESIGN,
    CONTRACT_UNRESOLVED,
)

IMPLEMENTATION_PASSING = "PASSING"
IMPLEMENTATION_KNOWN_GAP = "KNOWN_GAP"
IMPLEMENTATION_REFUSES_CORRECTLY = "REFUSES_CORRECTLY"
IMPLEMENTATION_TEST_GAP = "TEST_GAP"
IMPLEMENTATION_UNMEASURED = "UNMEASURED"
IMPLEMENTATION_STATUSES = (
    IMPLEMENTATION_PASSING,
    IMPLEMENTATION_KNOWN_GAP,
    IMPLEMENTATION_REFUSES_CORRECTLY,
    IMPLEMENTATION_TEST_GAP,
    IMPLEMENTATION_UNMEASURED,
)

# --------------------------------------------------------------------------
# How precisely the contributor can be named (Rev2 R4, provenance honesty)
# --------------------------------------------------------------------------

#: The registry value differs from the shipped default, so the registry was
#: demonstrably changed - though not by which of the possible writers.
ORIGIN_DETAIL_CHANGED = "CHANGED_FROM_DEFAULT"

#: The registry value equals the shipped default. Untouched and
#: "set back to the default" are indistinguishable, because the registry keeps
#: effective values and not their history. Reported as UNKNOWN rather than
#: guessed: claiming BUILTIN_DEFAULT here would be wrong whenever a caller
#: really did set the value.
ORIGIN_DETAIL_UNKNOWN = "UNKNOWN"

ORIGIN_DETAILS = (ORIGIN_DETAIL_CHANGED, ORIGIN_DETAIL_UNKNOWN)


# --------------------------------------------------------------------------
# One described field
# --------------------------------------------------------------------------

@dataclass
class Field:
    """One effective value, with where it came from.

    Attributes
    ----------
    path
        Where this sits in the description, for example ``statistic.bins``.
    value
        The effective value. This is what the drawing code will use.
    origin
        One of the origin constants above.
    config_key
        The registry key consulted, when one was.
    explicit_by_user
        Only meaningful when ``origin_sensitive`` is true. Records whether the
        caller supplied the value, because for those fields that fact changes
        behaviour and not merely bookkeeping.
    origin_sensitive
        True when explicitness is part of the meaning rather than a diagnostic
        detail.
    """

    path: str
    value: Any
    origin: str
    config_key: Optional[str] = None
    origin_detail: Optional[str] = None
    explicit_by_user: Optional[bool] = None
    origin_sensitive: bool = False

    def __post_init__(self):
        if self.origin not in ORIGINS:
            raise ValueError(
                f"unknown origin {self.origin!r} for {self.path!r}; "
                f"expected one of {ORIGINS}"
            )
        if (self.origin_detail is not None
                and self.origin_detail not in ORIGIN_DETAILS):
            raise ValueError(
                f"unknown origin_detail {self.origin_detail!r} for "
                f"{self.path!r}; expected one of {ORIGIN_DETAILS}"
            )

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"value": self.value, "source": self.origin}
        if self.origin_detail is not None:
            out["origin_detail"] = self.origin_detail
        if self.config_key is not None:
            out["config_key"] = self.config_key
        if self.origin_sensitive:
            out["explicit_by_user"] = bool(self.explicit_by_user)
        return out


# --------------------------------------------------------------------------
# The recorder — carried through resolution, or absent
# --------------------------------------------------------------------------

class Description:
    """Collects described fields in the order they are resolved.

    A `Description` is optional everywhere. When production code runs normally
    it passes ``None`` and nothing is recorded, so there is no cost and no
    behaviour change. When an explanation is being produced, the same code runs
    with a `Description` attached and the origins are captured on the way past.

    This is what keeps one implementation and two callers: the resolution is
    not duplicated, it is merely observed.
    """

    def __init__(self, *, view: str = "effective", door: str = "draw") -> None:
        self._fields: Dict[str, Field] = {}
        self._semantic_values: Dict[str, Any] = {}
        self._meta: Dict[str, Any] = {
            "view": view,
            "door": door,
            "contract_status": CONTRACT_SUPPORTED,
            "implementation_status": IMPLEMENTATION_PASSING,
            "evidence": [],
        }

    def record(self, f: Field) -> Field:
        self._fields[f.path] = f
        return f

    def record_semantic(self, path: str, value: Any) -> Any:
        """Record a semantic fact that has no configuration provenance.

        Examples are branch/group coordinates and contract lowering notes.
        These are deliberately separate from :class:`Field`, whose provenance
        vocabulary describes effective configuration values.
        """
        self._semantic_values[path] = value
        return value

    def set_status(
        self,
        *,
        contract_status: Optional[str] = None,
        implementation_status: Optional[str] = None,
        evidence: Optional[List[str]] = None,
    ) -> None:
        if contract_status is not None:
            if contract_status not in CONTRACT_STATUSES:
                raise ValueError(
                    f"unknown contract_status {contract_status!r}; "
                    f"expected one of {CONTRACT_STATUSES}"
                )
            self._meta["contract_status"] = contract_status
        if implementation_status is not None:
            if implementation_status not in IMPLEMENTATION_STATUSES:
                raise ValueError(
                    f"unknown implementation_status {implementation_status!r}; "
                    f"expected one of {IMPLEMENTATION_STATUSES}"
                )
            self._meta["implementation_status"] = implementation_status
        if evidence is not None:
            self._meta["evidence"] = list(evidence)

    def get(self, path: str) -> Optional[Field]:
        return self._fields.get(path)

    def get_semantic(self, path: str, default=None):
        return self._semantic_values.get(path, default)

    def fields(self) -> List[Field]:
        return list(self._fields.values())

    @staticmethod
    def _set_nested(out: Dict[str, Any], path: str, value: Any) -> None:
        node = out
        parts = path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def as_dict(self) -> Dict[str, Any]:
        """Nested dictionary keyed by semantic dotted paths."""
        out: Dict[str, Any] = {"_semantic": dict(self._meta)}
        for path, value in self._semantic_values.items():
            self._set_nested(out, path, value)
        for f in self._fields.values():
            self._set_nested(out, f.path, f.as_dict())
        return out

    def pretty(self) -> str:
        """Human-readable rendering of the same structured description."""
        lines: List[str] = [
            "SEMANTIC",
            f"  view                   {self._meta['view']}",
            f"  door                   {self._meta['door']}",
            f"  contract_status        {self._meta['contract_status']}",
            f"  implementation_status  {self._meta['implementation_status']}",
        ]
        for evidence in self._meta.get("evidence", []):
            lines.append(f"  evidence               {evidence}")
        lines.append("")

        by_section: Dict[str, List[Field]] = {}
        for f in self._fields.values():
            by_section.setdefault(f.path.split(".")[0], []).append(f)

        semantic_by_section: Dict[str, List[tuple]] = {}
        for path, value in self._semantic_values.items():
            semantic_by_section.setdefault(path.split(".")[0], []).append(
                (path, value)
            )

        for section in sorted(set(by_section) | set(semantic_by_section)):
            lines.append(section.upper())
            for path, value in semantic_by_section.get(section, []):
                leaf = path.split(".", 1)[1] if "." in path else path
                lines.append(f"  {leaf:<22} {value!r}")
            for f in by_section.get(section, []):
                leaf = f.path.split(".", 1)[1] if "." in f.path else f.path
                lines.append(f"  {leaf:<22} {f.value!r}")
                line = f"      source            {f.origin}"
                if f.config_key:
                    line += f"   [{f.config_key}]"
                lines.append(line)
                if f.origin_detail is not None:
                    lines.append(f"      origin_detail     {f.origin_detail}")
                if f.origin_sensitive:
                    lines.append(
                        f"      explicit_by_user  {bool(f.explicit_by_user)}"
                    )
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# The single resolution helper — used by production AND by explain
# --------------------------------------------------------------------------

def resolve(
    supplied: Any,
    *,
    path: str,
    config_key: Optional[str] = None,
    fallback: Any = None,
    description: Optional[Description] = None,
    origin_sensitive: bool = False,
    getter: Optional[Callable[[str, Any], Any]] = None,
    default_registry: Optional[Dict[str, Any]] = None,
) -> Any:
    """Resolve one value the way dfdraw already does, and record the origin.

    This reproduces the existing pattern exactly::

        if supplied is None:
            supplied = get_style_value(config_key, fallback)

    The returned value is therefore identical to what the previous inline code
    produced. The only addition is that, when a `Description` is supplied, the
    origin is written down.

    Parameters
    ----------
    supplied
        What the caller passed; ``None`` means "not supplied".
    path
        Dotted position in the description, e.g. ``statistic.bins``.
    config_key
        Registry key to consult when nothing was supplied. ``None`` means this
        field has no registry entry and the fallback applies directly.
    fallback
        Used when neither the caller nor the registry provides a value. This is
        the same literal the inline call site used.
    description
        Optional recorder. When ``None`` nothing is recorded and behaviour is
        exactly as before.
    origin_sensitive
        Set for fields whose explicitness changes behaviour.
    getter, default_registry
        Injection points for testing; production leaves them ``None`` and the
        live registry is used.
    """
    if getter is None:
        from ..style import get_style_value as getter  # type: ignore

    explicit = supplied is not None

    detail = None
    if explicit:
        value, origin, used_key = supplied, CALL_ARGUMENT, None
    elif config_key is None:
        value, origin, used_key = fallback, BUILTIN_DEFAULT, None
    else:
        value = getter(config_key, fallback)
        used_key = config_key
        origin, detail = _classify_global(
            config_key, value, fallback, default_registry)

    if description is not None:
        description.record(Field(
            path=path,
            value=value,
            origin=origin,
            config_key=used_key,
            origin_detail=detail,
            explicit_by_user=explicit if origin_sensitive else None,
            origin_sensitive=origin_sensitive,
        ))
    return value


def _classify_global(config_key, value, fallback, default_registry):
    """Name the source, and say how precisely the contributor is known.

    Returns ``(origin, origin_detail)``. The registry keeps effective values
    but not their history, so a value equal to the shipped default may mean
    "nobody changed it" or "somebody set it back". Rev2 requires reporting
    the source as CURRENT_GLOBAL_CONFIGURATION with origin_detail=UNKNOWN
    rather than guessing BUILTIN_DEFAULT.
    """
    if default_registry is None:
        try:
            from ..style import DEFAULT_STYLE as default_registry  # type: ignore
        except Exception:
            default_registry = None

    if default_registry is None or config_key not in default_registry:
        # No registry entry at all: the call-site fallback is what applied.
        if value == fallback:
            return BUILTIN_DEFAULT, None
        return CURRENT_GLOBAL_CONFIGURATION, ORIGIN_DETAIL_CHANGED

    shipped = default_registry[config_key]
    if value == shipped:
        return CURRENT_GLOBAL_CONFIGURATION, ORIGIN_DETAIL_UNKNOWN
    return CURRENT_GLOBAL_CONFIGURATION, ORIGIN_DETAIL_CHANGED


def describe_supplied(**supplied: Any) -> Dict[str, Any]:
    """The SUPPLIED view: only what the caller actually wrote.

    Deliberately trivial — it must not resolve anything, or it stops being a
    record of what was asked for.
    """
    return {k: v for k, v in supplied.items() if v is not None}


def record_supplied_fields(specs, supplied, description):
    """Record only explicitly supplied fields using the canonical specs.

    This is the SUPPLIED-view counterpart of :func:`resolve_fields`: it reuses
    the same declaration owner but intentionally performs no configuration or
    default resolution.
    """
    for spec in specs:
        if spec.arg in supplied and supplied[spec.arg] is not None:
            description.record(Field(
                path=spec.path,
                value=supplied[spec.arg],
                origin=CALL_ARGUMENT,
                explicit_by_user=True if spec.origin_sensitive else None,
                origin_sensitive=spec.origin_sensitive,
            ))
    return description


# --------------------------------------------------------------------------
# Field declarations — the single owner of WHAT is resolved
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    """One field's contract: which argument, which path, which key, what default.

    PHASE_13_82_DF review finding P1-1. `resolve()` owns HOW a value is
    resolved; this owns WHAT is being resolved and under which contract. Before
    this existed, the production path and the explain surface each wrote their
    own `path` / `config_key` / `fallback` / `origin_sensitive` for the same
    field, so a change to one - renaming a configuration key, moving a default,
    flipping origin-sensitivity - could leave the other stale. T11 would
    probably catch the drift, but the point of this phase is to remove the
    duplication rather than to detect it after the fact.

    Attributes
    ----------
    arg
        The public keyword this field arrives as, e.g. ``bins``.
    path
        Where it sits in the description, e.g. ``statistic.bins``.
    config_key
        Registry key consulted when nothing was supplied; ``None`` when the
        field has no registry entry.
    fallback
        Used when neither caller nor registry supplies a value. Must equal the
        literal the original inline call site used.
    origin_sensitive
        True when explicitness itself changes behaviour, not merely bookkeeping.
    """

    arg: str
    path: str
    config_key: Optional[str]
    fallback: Any
    origin_sensitive: bool = False


#: The static (data-free) fields of a profile draw, in resolution order.
#: Consumed by BOTH `plots/profile.py:draw_profile` and `DFDraw._explain`.
#: Stage 1 scope: four fields. Deliberately not generalised to the other
#: resolution sites yet.
PROFILE_STATIC_FIELDS = (
    FieldSpec("bins", "statistic.bins", "hist.bins", 50),
    FieldSpec("marker", "aesthetics.marker", "profile.marker", "o",
              origin_sensitive=True),
    FieldSpec("markersize", "aesthetics.markersize", "profile.markersize", 6,
              origin_sensitive=True),
    FieldSpec("capsize", "aesthetics.capsize", "profile.capsize", 3),
)


def resolve_fields(specs, supplied, description=None, **resolve_kwargs):
    """Resolve a whole declaration set, returning ``{arg: value}``.

    Both callers use this, so neither can hold a stale copy of a field's
    contract.

    Parameters
    ----------
    specs
        Iterable of `FieldSpec`, e.g. `PROFILE_STATIC_FIELDS`.
    supplied
        Mapping of public keyword to what the caller passed; a missing key or
        a ``None`` value both mean "not supplied".
    description
        Optional recorder; ``None`` on every normal draw.
    """
    out = {}
    for spec in specs:
        out[spec.arg] = resolve(
            supplied.get(spec.arg),
            path=spec.path,
            config_key=spec.config_key,
            fallback=spec.fallback,
            description=description,
            origin_sensitive=spec.origin_sensitive,
            **resolve_kwargs,
        )
    return out
