"""Pure 2-D facet interval display formatting for PHASE_13_97_DF.

This module owns *display strings only*.  It must never alter pandas Interval
objects, bin edges, row membership, result keys, or plotting/statistical data.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numbers
from typing import Any, Dict, Iterable, List

import pandas as pd


_RESERVED = frozenset({"format", "by"})


def _normalize_policy(policy: Any) -> str:
    """Normalize one scalar display policy to its canonical string form."""
    if policy is None or policy == "legacy":
        return "legacy"
    if not isinstance(policy, str):
        raise TypeError(
            "facet label format policy must be None, 'legacy', 'auto', "
            "or a Python numeric format string")
    if policy == "auto":
        return policy
    if not policy:
        raise ValueError("facet label numeric format string must not be empty")
    try:
        format(1.2345, policy)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid facet label numeric format {policy!r}: {exc}") from exc
    return policy


def normalize_facet_label_format(value: Any) -> Dict[str, Any]:
    """Normalize public/style shorthand to ``{'format': ..., 'by': ...}``.

    Plain mappings are atomic per-dimension replacements with a legacy general
    fallback.  A mapping containing either reserved structural key (``format``
    or ``by``) is interpreted as the canonical full form and may contain no
    other top-level keys.
    """
    if value is None or isinstance(value, str):
        return {"format": _normalize_policy(value), "by": {}}
    if not isinstance(value, Mapping):
        raise TypeError(
            "facet_label_format must be None, a scalar policy string, or a mapping")

    keys = set(value)
    if keys & _RESERVED:
        extra = keys - _RESERVED
        if extra:
            raise ValueError(
                "structured facet_label_format accepts only reserved keys "
                f"'format' and 'by'; unexpected keys: {sorted(map(str, extra))}")
        general = _normalize_policy(value.get("format", "legacy"))
        by = value.get("by", {})
        if by is None:
            by = {}
        if not isinstance(by, Mapping):
            raise TypeError("facet_label_format['by'] must be a mapping")
        normalized_by = {}
        for name, policy in by.items():
            if not isinstance(name, str) or not name:
                raise TypeError("facet_label_format dimension names must be non-empty strings")
            normalized_by[name] = _normalize_policy(policy)
        return {"format": general, "by": normalized_by}

    normalized_by = {}
    for name, policy in value.items():
        if not isinstance(name, str) or not name:
            raise TypeError("facet_label_format dimension names must be non-empty strings")
        normalized_by[name] = _normalize_policy(policy)
    return {"format": "legacy", "by": normalized_by}


def validate_facet_label_dimensions(spec: Mapping[str, Any], dimensions: Sequence[str]) -> None:
    """Reject per-dimension overrides that do not name an active facet."""
    active = set(dimensions)
    unknown = sorted(set(spec.get("by", {})) - active)
    if unknown:
        raise ValueError(
            f"facet_label_format contains unknown facet dimension(s) {unknown!r}; "
            f"active facet_by={list(dimensions)!r}")


def facet_label_format_is_legacy(spec: Mapping[str, Any]) -> bool:
    """True iff a normalized spec requests no non-legacy formatting."""
    if spec.get("format", "legacy") != "legacy":
        return False
    return all(v == "legacy" for v in spec.get("by", {}).values())


def _is_numeric_endpoint(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, (bool,))


def _closed_brackets(interval: pd.Interval) -> tuple[str, str]:
    left = "[" if interval.closed in ("left", "both") else "("
    right = "]" if interval.closed in ("right", "both") else ")"
    return left, right


def _format_number(value: float, spec: str) -> str:
    # Literal negative zero is a display artefact.  Only normalize it when the
    # underlying numeric endpoint is actually equal to zero.
    v = 0.0 if value == 0 else value
    return format(v, spec)


def _endpoint_map_explicit(endpoints: Iterable[float], spec: str) -> Dict[float, str]:
    ordered = list(dict.fromkeys(float(v) for v in endpoints))
    texts = {v: _format_number(v, spec) for v in ordered}
    reverse: Dict[str, float] = {}
    for value in ordered:
        text = texts[value]
        if text in reverse and reverse[text] != value:
            raise ValueError(
                "distinct facet endpoints collide under explicit format "
                f"{spec!r}: {reverse[text]!r} and {value!r} both display as "
                f"{text!r}; use 'auto' or a more precise format")
        reverse[text] = value
    return texts


def _endpoint_map_auto(endpoints: Iterable[float]) -> Dict[float, str]:
    ordered = list(dict.fromkeys(float(v) for v in endpoints))
    if not ordered:
        return {}
    for precision in range(3, 18):
        spec = f".{precision}g"
        texts = {v: _format_number(v, spec) for v in ordered}
        if len(set(texts.values())) == len(ordered):
            return texts
    # ``.17g`` should distinguish finite binary64 values.  Keep a faithful
    # fallback for completeness rather than ever changing an endpoint.
    texts = {v: repr(0.0 if v == 0 else float(v)) for v in ordered}
    if len(set(texts.values())) != len(ordered):
        raise ValueError("unable to build unique faithful facet endpoint labels")
    return texts


def format_facet_interval_dimension(values: Sequence[Any], policy: str) -> List[str]:
    """Format one complete facet dimension with one shared endpoint map.

    Non-interval categorical values are intentionally unchanged under every
    numeric policy.  Non-numeric Interval endpoints are outside Stage A and are
    rejected only when non-legacy formatting is explicitly requested.
    """
    policy = _normalize_policy(policy)
    if policy == "legacy":
        return [str(v) for v in values]

    intervals = [v for v in values if isinstance(v, pd.Interval)]
    if not intervals:
        return [str(v) for v in values]
    if len(intervals) != len(values):
        # A facet dimension should not mix interval and categorical identities;
        # keeping categorical text exact is safer than guessing a mixed grammar.
        return [str(v) for v in values]
    if any(not (_is_numeric_endpoint(v.left) and _is_numeric_endpoint(v.right))
           for v in intervals):
        raise TypeError(
            "non-legacy facet_label_format supports numeric pandas Interval "
            "facet dimensions only")

    endpoints = []
    for interval in intervals:
        endpoints.extend((float(interval.left), float(interval.right)))
    endpoint_map = (_endpoint_map_auto(endpoints) if policy == "auto"
                    else _endpoint_map_explicit(endpoints, policy))

    out = []
    for interval in intervals:
        lb, rb = _closed_brackets(interval)
        left = endpoint_map[float(interval.left)]
        right = endpoint_map[float(interval.right)]
        out.append(f"{lb}{left}, {right}{rb}")
    return out


def format_facet_dimension(values: Sequence[Any], dimension: str,
                           spec: Mapping[str, Any]) -> List[str]:
    """Apply the effective policy for one named facet dimension."""
    policy = spec.get("by", {}).get(dimension, spec.get("format", "legacy"))
    return format_facet_interval_dimension(values, policy)
