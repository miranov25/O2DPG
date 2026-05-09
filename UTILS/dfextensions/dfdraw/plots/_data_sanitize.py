"""
Data sanitization for dfdraw plots — Phase 13.28.DF.

NaN/inf handling with optional nan_policy parameter and counter reporting.
Centralizes sanitization across hist / hist2d / hexbin / profile / scatter.

References
----------
- Proposal: PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md §3, §7.1
- AD-69 (centralized sanitization module)
- AD-70 (nan_policy default 'filter')
- AD-71 (counters always populated)
"""
from typing import Optional, Tuple, Dict, Any
import warnings
import numpy as np


_VALID_NAN_POLICIES = ("filter", "warn", "raise")


def sanitize_for_plot(
    x_data: np.ndarray,
    y_data: Optional[np.ndarray] = None,
    nan_policy: str = "filter",
    column_names: Tuple[str, str] = ("x", "y"),
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Apply NaN/inf handling per nan_policy.

    Parameters
    ----------
    x_data : np.ndarray
        Input x values (may contain NaN, inf).
    y_data : np.ndarray, optional
        Input y values for 2D plots; None for 1D.
    nan_policy : {'filter', 'warn', 'raise'}, default 'filter'
        - 'filter' : drop NaN/inf silently; counters in returned stats.
                     If ALL rows dropped from non-empty input, emit one
                     UserWarning explaining why.
        - 'warn'   : drop + emit UserWarning with details whenever any
                     non-finite values are filtered.
        - 'raise'  : raise ValueError if any NaN/inf present.
    column_names : tuple of str
        Column labels for messages. Defaults to ('x', 'y').

    Returns
    -------
    x_clean : np.ndarray
        Sanitized x data.
    y_clean : np.ndarray or None
        Sanitized y data (None if y_data was None).
    sanitize_stats : dict
        Always populated with keys:
            n_input    — rows in x_data before sanitization
            n_filtered — rows removed (n_input - len(x_clean))
            n_inf_x, n_nan_x — counts in x column
            n_inf_y, n_nan_y — counts in y column (0 for 1D)

    Raises
    ------
    ValueError
        - When nan_policy not in {'filter', 'warn', 'raise'}.
        - When nan_policy='raise' AND any NaN/inf present (with column-named message).
    """
    if nan_policy not in _VALID_NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {_VALID_NAN_POLICIES}, got {nan_policy!r}"
        )

    # Force ndarray (caller may pass list / pandas Series)
    x_data = np.asarray(x_data)

    n_input = len(x_data)
    n_inf_x = int(np.isinf(x_data).sum())
    n_nan_x = int(np.isnan(x_data).sum())

    if y_data is not None:
        y_data = np.asarray(y_data)
        if len(y_data) != n_input:
            raise ValueError(
                f"x_data and y_data must have same length, got {n_input} vs {len(y_data)}"
            )
        n_inf_y = int(np.isinf(y_data).sum())
        n_nan_y = int(np.isnan(y_data).sum())
    else:
        n_inf_y = 0
        n_nan_y = 0

    has_invalid = bool(n_inf_x or n_nan_x or n_inf_y or n_nan_y)

    if has_invalid and nan_policy == "raise":
        raise ValueError(
            f"Cannot proceed: input has invalid values.\n"
            f"  Column '{column_names[0]}': {n_inf_x} inf, {n_nan_x} NaN\n"
            f"  Column '{column_names[1]}': {n_inf_y} inf, {n_nan_y} NaN\n"
            f"  Use nan_policy='filter' (default) to drop invalid rows, "
            f"or fix the data source."
        )

    # Build mask. For 1D: just x. For 2D: x AND y must both be finite.
    mask = np.isfinite(x_data)
    if y_data is not None:
        mask = mask & np.isfinite(y_data)

    n_finite = int(mask.sum())

    if has_invalid and nan_policy == "warn":
        warnings.warn(
            f"sanitize_for_plot: filtered {n_input - n_finite} non-finite "
            f"rows from {column_names}. "
            f"'{column_names[0]}': {n_inf_x} inf, {n_nan_x} NaN; "
            f"'{column_names[1]}': {n_inf_y} inf, {n_nan_y} NaN.",
            UserWarning, stacklevel=2,
        )

    x_clean = x_data[mask]
    y_clean = y_data[mask] if y_data is not None else None

    # Suspicious: input had data but sanitize dropped everything.
    # Warn under 'filter' (the default) — under 'warn' we already warned above.
    if (
        nan_policy == "filter"
        and n_finite == 0
        and n_input > 0
    ):
        warnings.warn(
            f"sanitize_for_plot: all {n_input} rows dropped after sanitization. "
            f"'{column_names[0]}': {n_inf_x} inf + {n_nan_x} NaN; "
            f"'{column_names[1]}': {n_inf_y} inf + {n_nan_y} NaN. "
            f"Returning empty arrays.",
            UserWarning, stacklevel=2,
        )

    sanitize_stats: Dict[str, Any] = {
        "n_input":    n_input,
        "n_filtered": n_input - n_finite,
        "n_inf_x":    n_inf_x,
        "n_nan_x":    n_nan_x,
        "n_inf_y":    n_inf_y,
        "n_nan_y":    n_nan_y,
    }

    return x_clean, y_clean, sanitize_stats
