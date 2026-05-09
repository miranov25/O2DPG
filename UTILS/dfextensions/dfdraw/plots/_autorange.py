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
