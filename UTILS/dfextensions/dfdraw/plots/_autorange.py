"""
Autorange strategies for dfdraw plots — Phase 13.28.DF Part B.

Hybrid + minmax + percentile + robust strategies for outlier-aware autorange.
Used by hist / hist2d / hexbin / profile / scatter when range=None or when
range is a strategy string ('hybrid', 'minmax', 'percentile_99', etc.).

References
----------
- Proposal: PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md §4
- AD-72 (hybrid algorithm formal definition)
- AD-73 (default strategy 'hybrid')
- AD-74 (2D autorange per-axis independent)
- AD-77 (autorange_used + autorange_strategy in stats dict)
"""
from typing import Tuple
import numpy as np


# Valid strategy preset names (AD-73)
VALID_STRATEGIES = (
    "minmax",         # backward compat / clean data
    "hybrid",         # default — outlier-aware
    "percentile_99",  # 1st-99th percentile clip
    "percentile_95",  # 2.5th-97.5th percentile clip
    "robust_3mad",    # median ± 3·sigma_MAD (no outlier check)
    "robust_4mad",    # median ± 4·sigma_MAD (no outlier check)
)


def hybrid_autorange(
    data: np.ndarray,
    k_robust: float = 4.0,
    k_outlier: float = 1.5,
) -> Tuple[float, float]:
    """
    Outlier-aware autorange combining robust window and (min, max).

    Computes robust window (median ± k_robust·sigma_MAD). For each side
    independently, declares outlier if data extreme exceeds median by more
    than k_outlier·k_robust·sigma_MAD. Per-side: uses robust bound when
    outlier present; uses data extreme when no outlier.

    See PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md §4.1 for full
    algorithm and §4.2 for worked examples.

    Parameters
    ----------
    data : np.ndarray
        Finite 1D data (post-sanitization). Caller is responsible for
        filtering NaN/inf via sanitize_for_plot before calling this.
    k_robust : float, default 4.0
        Half-width of robust window in MAD-equivalent sigmas.
        4.0 covers ~99.99% of a Gaussian distribution.
    k_outlier : float, default 1.5
        Tolerance multiplier for outlier declaration on each side.

    Returns
    -------
    (lo, hi) : tuple of float
        Per-side decision result.
    """
    data = np.asarray(data)
    if len(data) == 0:
        return (0.0, 1.0)

    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    sigma_mad = 1.4826 * mad   # MAD → Gaussian-equivalent sigma

    if sigma_mad == 0.0:
        # All values equal — degenerate; return narrow window centered on median
        return (median - 0.5, median + 0.5)

    robust_lo = median - k_robust * sigma_mad
    robust_hi = median + k_robust * sigma_mad
    threshold_lo = median - k_outlier * k_robust * sigma_mad
    threshold_hi = median + k_outlier * k_robust * sigma_mad

    data_min = float(data.min())
    data_max = float(data.max())

    has_outlier_lo = data_min < threshold_lo
    has_outlier_hi = data_max > threshold_hi

    lo = robust_lo if has_outlier_lo else data_min
    hi = robust_hi if has_outlier_hi else data_max

    return (float(lo), float(hi))


def _minmax_autorange(data: np.ndarray) -> Tuple[float, float]:
    """Plain (min, max) — matplotlib-equivalent, backward compat."""
    data = np.asarray(data)
    if len(data) == 0:
        return (0.0, 1.0)
    return (float(data.min()), float(data.max()))


def _percentile_autorange(
    data: np.ndarray, low: float, high: float
) -> Tuple[float, float]:
    """Bounds at low/high percentiles. low and high in [0, 100]."""
    data = np.asarray(data)
    if len(data) == 0:
        return (0.0, 1.0)
    return (float(np.percentile(data, low)), float(np.percentile(data, high)))


def _robust_autorange(
    data: np.ndarray, k_robust: float
) -> Tuple[float, float]:
    """Plain robust window: median ± k·sigma_MAD. No outlier check."""
    data = np.asarray(data)
    if len(data) == 0:
        return (0.0, 1.0)
    median = float(np.median(data))
    mad = float(np.median(np.abs(data - median)))
    sigma_mad = 1.4826 * mad
    if sigma_mad == 0.0:
        return (median - 0.5, median + 0.5)
    return (median - k_robust * sigma_mad, median + k_robust * sigma_mad)


def compute_autorange(
    data: np.ndarray,
    strategy: str = "hybrid",
    k_robust: float = 4.0,
    k_outlier: float = 1.5,
    percentile: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[float, float]:
    """
    Compute autorange for 1D data per strategy preset.

    Parameters
    ----------
    data : np.ndarray
        Finite 1D data (post-sanitization).
    strategy : str, default 'hybrid'
        One of VALID_STRATEGIES.
    k_robust : float, default 4.0
        For 'hybrid' and 'robust_*mad' (where the value is overridden by the
        preset name suffix).
    k_outlier : float, default 1.5
        For 'hybrid' only.
    percentile : (float, float), default (1.0, 99.0)
        For 'percentile_99' (default uses this); 'percentile_95' overrides.

    Returns
    -------
    (lo, hi) : tuple of float
        Numeric range. Caller stores in stats['autorange_used'] (AD-77).

    Raises
    ------
    ValueError
        When strategy is not in VALID_STRATEGIES.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"strategy must be one of {VALID_STRATEGIES}, got {strategy!r}"
        )

    # Phase 13.51 §1.2.3 (audit S-4 V-2): datetime64 guard at single entry
    # covers all 5 strategy paths. np.median/np.percentile/data.min on
    # datetime64 arrays return timedelta64 which float() cannot convert.
    # Mirror the pattern at plots/profile.py:453-468 used in non-faceted path.
    data = np.asarray(data)
    if np.issubdtype(data.dtype, np.datetime64):
        data = data.astype('datetime64[s]').astype(np.int64).astype(float)

    if strategy == "minmax":
        return _minmax_autorange(data)

    if strategy == "hybrid":
        return hybrid_autorange(data, k_robust=k_robust, k_outlier=k_outlier)

    if strategy == "percentile_99":
        return _percentile_autorange(data, percentile[0], percentile[1])

    if strategy == "percentile_95":
        return _percentile_autorange(data, 2.5, 97.5)

    if strategy == "robust_3mad":
        return _robust_autorange(data, k_robust=3.0)

    if strategy == "robust_4mad":
        return _robust_autorange(data, k_robust=4.0)

    # Should be unreachable — caught by the VALID_STRATEGIES check above
    raise ValueError(f"Unknown strategy: {strategy!r}")


def resolve_range_1d(
    range_arg,
    data: np.ndarray,
    style_strategy: str = "hybrid",
    style_k_robust: float = 4.0,
    style_k_outlier: float = 1.5,
    style_percentile: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[Tuple[float, float], str]:
    """
    Resolve a `range=` parameter for 1D plots into (used_range, strategy_label).

    Accepts:
        None                         → use style_strategy
        'auto'                       → use style_strategy
        '<strategy>' (string)        → use named strategy from VALID_STRATEGIES
        (lo, hi) tuple of numbers    → use as-is, label='explicit'

    Parameters
    ----------
    range_arg : None | str | (float, float)
        User-supplied range argument.
    data : np.ndarray
        Finite 1D data (post-sanitization). Used when computing autorange.
    style_strategy : str
        Default strategy when range_arg is None or 'auto'.
    style_k_robust, style_k_outlier, style_percentile :
        Tunable parameters from style keys.

    Returns
    -------
    (lo, hi) : tuple of float
    strategy_label : str
        Strategy actually used (for stats['autorange_strategy']).
        'explicit' when user passed numeric range.
    """
    # Explicit numeric range: pass through
    if isinstance(range_arg, (tuple, list)) and not isinstance(range_arg, str):
        if len(range_arg) == 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in range_arg):
            return (float(range_arg[0]), float(range_arg[1])), "explicit"

    # None or 'auto' → use style default
    if range_arg is None or range_arg == "auto":
        strategy = style_strategy
    elif isinstance(range_arg, str):
        strategy = range_arg
    else:
        # Unknown form (not numeric tuple, not string) — let it raise downstream
        raise ValueError(
            f"range must be None, 'auto', a strategy name from {VALID_STRATEGIES}, "
            f"or a (lo, hi) tuple. Got {range_arg!r}."
        )

    used = compute_autorange(
        data,
        strategy=strategy,
        k_robust=style_k_robust,
        k_outlier=style_k_outlier,
        percentile=style_percentile,
    )
    return used, strategy


def resolve_range_2d(
    range_arg,
    x_data: np.ndarray,
    y_data: np.ndarray,
    style_strategy: str = "hybrid",
    style_k_robust: float = 4.0,
    style_k_outlier: float = 1.5,
    style_percentile: Tuple[float, float] = (1.0, 99.0),
) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], str]:
    """
    Resolve a `range=` parameter for 2D plots (per-axis independent — AD-74).

    Accepts:
        None                                   → use style_strategy per-axis
        'auto'                                 → use style_strategy per-axis
        '<strategy>' (string)                  → use named strategy per-axis
        ((xlo, xhi), (ylo, yhi)) explicit      → pass through, label='explicit'

    Returns
    -------
    ((xlo, xhi), (ylo, yhi)) : tuple of tuples
    strategy_label : str
    """
    # Explicit ((xlo, xhi), (ylo, yhi)): pass through
    if isinstance(range_arg, (tuple, list)) and not isinstance(range_arg, str):
        if (
            len(range_arg) == 2
            and all(
                isinstance(side, (tuple, list)) and len(side) == 2
                and all(isinstance(v, (int, float, np.integer, np.floating)) for v in side)
                for side in range_arg
            )
        ):
            return (
                ((float(range_arg[0][0]), float(range_arg[0][1])),
                 (float(range_arg[1][0]), float(range_arg[1][1]))),
                "explicit",
            )

    # None / 'auto' → use style; '<strategy>' → use named strategy
    if range_arg is None or range_arg == "auto":
        strategy = style_strategy
    elif isinstance(range_arg, str):
        strategy = range_arg
    else:
        raise ValueError(
            f"range for 2D must be None, 'auto', a strategy name from {VALID_STRATEGIES}, "
            f"or ((xlo,xhi),(ylo,yhi)). Got {range_arg!r}."
        )

    # AD-74: per-axis independent autorange
    x_range = compute_autorange(
        x_data, strategy=strategy,
        k_robust=style_k_robust, k_outlier=style_k_outlier, percentile=style_percentile,
    )
    y_range = compute_autorange(
        y_data, strategy=strategy,
        k_robust=style_k_robust, k_outlier=style_k_outlier, percentile=style_percentile,
    )
    return (x_range, y_range), strategy
