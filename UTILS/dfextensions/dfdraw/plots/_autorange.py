"""
Autorange strategies for dfdraw plots — Phase 13.28.DF (Commit 1 scaffolding).

Hybrid + minmax + percentile + robust strategies for outlier-aware autorange.
Used by hist / hist2d / hexbin / profile / scatter when range=None or when
range is a strategy string ('hybrid', 'minmax', 'percentile_99', etc.).

This is the Commit 1 scaffold — function bodies raise NotImplementedError.
Commit 2b (Part B) populates the bodies and integrates into plot modules.

References
----------
- Proposal: PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md §4, §7.1
- AD-72 (hybrid algorithm formal definition)
- AD-73 (default strategy 'hybrid')
- AD-74 (2D autorange per-axis independent)
- AD-77 (autorange_used + autorange_strategy in stats dict)
"""
from typing import Tuple
import numpy as np


# Valid strategy preset names (AD-73)
VALID_STRATEGIES = (
    "minmax",        # backward compat / clean data
    "hybrid",        # default — outlier-aware
    "percentile_99", # 1st-99th percentile clip
    "percentile_95", # 2.5th-97.5th percentile clip
    "robust_3mad",   # median ± 3·sigma_MAD (no outlier check)
    "robust_4mad",   # median ± 4·sigma_MAD (no outlier check)
)


def compute_autorange(
    data: np.ndarray,
    strategy: str = "hybrid",
    **strategy_kwargs,
) -> Tuple[float, float]:
    """
    Compute autorange for 1D data per strategy preset.

    Parameters
    ----------
    data : np.ndarray
        Finite 1D data (post-sanitization).
    strategy : str, default 'hybrid'
        One of VALID_STRATEGIES. Style key 'autorange.strategy' provides
        the per-process default; per-call override via this argument or
        via the plot's `range='<strategy>'` kwarg.
    **strategy_kwargs
        Strategy-specific parameters. v1.0 takes defaults from style keys
        (autorange.k_robust, autorange.k_outlier, autorange.percentile);
        per-call override deferred to Phase 13.29 per AD-76.

    Returns
    -------
    (lo, hi) : tuple of float
        Numeric range. Caller stores in stats['autorange_used'] (AD-77).

    Raises
    ------
    ValueError
        When strategy is not in VALID_STRATEGIES.
    NotImplementedError
        Phase 13.28.DF Commit 1 scaffold — body in Commit 2b.
    """
    raise NotImplementedError(
        f"compute_autorange (strategy={strategy!r}) — Phase 13.28.DF Commit 2b will populate. "
        "See PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md §4."
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
    algorithm and §4.2 for worked examples (clean Gaussian, single outlier,
    asymmetric tail, constant data).

    Parameters
    ----------
    data : np.ndarray
        Finite 1D data (post-sanitization).
    k_robust : float, default 4.0
        Half-width of robust window in MAD-equivalent sigmas.
        4.0 covers ~99.99% of a Gaussian distribution.
    k_outlier : float, default 1.5
        Tolerance multiplier for outlier declaration on each side.

    Returns
    -------
    (lo, hi) : tuple of float
        Per-side decision result.

    Raises
    ------
    NotImplementedError
        Phase 13.28.DF Commit 1 scaffold — body in Commit 2b.
    """
    raise NotImplementedError(
        "hybrid_autorange — Phase 13.28.DF Commit 2b will populate. "
        "See PHASE_13_28_DF_v1_1_Proposal_RobustDataHandling.md §4.1."
    )
