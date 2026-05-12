"""
Parameter validation for dfdraw plot calls — Phase 13.30.DF v1.0.

Class-2 column-reference validation: enforce that string-valued parameters
naming a DataFrame column refer to an existing column, instead of silently
falling through to ungrouped/unfiltered behaviour when the column is missing.

Background
----------
- BUG_ADF_GroupBy_Expression_Materialization (2026-05-11): ADF forwards
  `group_by="row%3"` unmaterialized; dfdraw's silent-fallthrough check
  (e.g. `if group_by in df.columns`) hid the bug behind a downstream
  matplotlib UserWarning about empty legend labels.

- Phase 13.30.DF sub-fix 1: introduce class-level tuple + helper so all
  Class-2 parameters are validated uniformly, and Phase 13.27 Commit 2's
  selection_vector / weights_vector get coverage for free when their
  tuples are populated.

Parameter taxonomy (see Phase 13.30 proposal §3 for the full table)
-------------------------------------------------------------------
- Class 2 — column reference (strict): string MUST be an existing column.
  This module handles Class 2. Examples: ``group_by``.

- Class 4 — expression-or-column (permissive): string is either column
  name OR pandas eval expression; dfdraw evaluates internally.
  Examples: ``weights`` (handled by ``_eval_weights`` in plots/profile.py).
  These parameters are NOT validated here.

References
----------
- BUG_ADF_GroupBy_Expression_Materialization.md
- PHASE_13_30_DF_v1_0_Proposal_ParameterClassValidation.md §3, §4
"""
from typing import Iterable, Mapping, Any, List, Tuple
import pandas as pd


def validate_column_references(
    df: pd.DataFrame,
    kwargs: Mapping[str, Any],
    names: Iterable[str],
    context: str,
) -> None:
    """Validate that every Class-2 column-reference kwarg names a real column.

    For each name in ``names``, look up the value in ``kwargs``. If the
    value is a non-empty string and is NOT a column in ``df``, raise
    ``ValueError`` with an actionable message naming the offending
    parameter, the bad value, the available columns (first 10), and a
    pointer to the materialize-first remediation.

    Class-2 parameters are strict column references — computed expressions
    must be materialized by the caller (e.g., via
    ``AliasDataFrame.add_alias``). Class-4 parameters (like ``weights``,
    which dfdraw evaluates via ``df.eval``) are NOT validated here and
    must NOT be listed in ``names``.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame the plot call will operate on.
    kwargs : Mapping[str, Any]
        Keyword arguments of the calling plot function. Typically pass
        ``locals()`` from the call site.
    names : Iterable[str]
        Names of Class-2 column-reference parameters to validate. Driven
        by the per-plot ``_*_COLUMN_REFERENCES`` tuple in ``drawer.py``.
        Pass the tuple directly to avoid duplication and drift.
    context : str
        Plot-type name (``"profile"`` / ``"hist"`` / ``"scatter"`` etc.).
        Used only to make the error message actionable.

    Raises
    ------
    ValueError
        If any value in ``names`` is a non-empty string not present in
        ``df.columns``. Single-bad case produces a per-parameter message;
        multi-bad case enumerates all offenders in one error.

    Notes
    -----
    - ``None`` values, non-string values, and the empty string are skipped
      (these are not the bug class we're catching — they have other paths).
    - Empty DataFrames (no columns at all) are passed through silently;
      downstream code already handles empty input.

    Examples
    --------
    >>> df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "sector": [0, 1, 0]})
    >>> # Existing column — no raise:
    >>> validate_column_references(df, {"group_by": "sector"},
    ...                            names=("group_by",), context="profile")

    >>> # Missing column — raises ValueError:
    >>> validate_column_references(df, {"group_by": "row%3"},
    ...                            names=("group_by",), context="profile")
    Traceback (most recent call last):
        ...
    ValueError: group_by='row%3' is not a column in the DataFrame ...
    """
    if df is None or len(getattr(df, "columns", [])) == 0:
        # No columns to check against; downstream code handles empty df
        return

    bad: List[Tuple[str, str]] = []
    for name in names:
        value = kwargs.get(name)
        if value is None or not isinstance(value, str) or not value:
            continue
        if value in df.columns:
            continue
        bad.append((name, value))

    if not bad:
        return

    cols = list(df.columns)
    if len(cols) > 10:
        cols_preview = ", ".join(repr(c) for c in cols[:10]) + f", ... ({len(cols)} total)"
    else:
        cols_preview = ", ".join(repr(c) for c in cols)

    if len(bad) == 1:
        name, value = bad[0]
        raise ValueError(
            f"{name}={value!r} is not a column in the DataFrame "
            f"(called from dfdraw {context}()). "
            f"If this is a computed expression, the caller must materialize "
            f"it into a column first. With AliasDataFrame: "
            f"adf.add_alias(<name>, {value!r}). "
            f"Available columns: [{cols_preview}]"
        )

    bad_str = ", ".join(f"{n}={v!r}" for n, v in bad)
    raise ValueError(
        f"{len(bad)} column-reference parameters do not name DataFrame "
        f"columns (called from dfdraw {context}()): {bad_str}. "
        f"Materialize computed expressions into columns first. "
        f"Available columns: [{cols_preview}]"
    )
