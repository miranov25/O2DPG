"""
Style management for dfdraw.

Supports:
- Predefined styles (default, publication, presentation)
- Custom style dictionaries
- JSON save/load for persistence
- Mapping to matplotlib rcParams
"""

import json
import copy
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Union

# =============================================================================
# Default Style Definition
# =============================================================================

DEFAULT_STYLE: Dict[str, Any] = {
    # Figure
    "figure.figsize": (8, 6),
    "figure.dpi": 100,
    "figure.facecolor": "white",
    
    # Fonts
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    
    # Grid
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    
    # Scatter
    "scatter.alpha": 0.7,
    "scatter.edgecolors": "black",
    "scatter.linewidths": 0.5,
    "scatter.size": 50,
    
    # Histogram
    "hist.bins": 50,
    "hist.alpha": 0.7,
    "hist.histtype": "stepfilled",
    "hist.edgecolor": "black",
    "hist.linewidth": 1.0,
    
    # Profile
    "profile.marker": "o",
    "profile.markersize": 6,
    "profile.capsize": 3,
    # Cap size for SEM/STD error bars on profile() central line. Independent of
    # quantile.error_bars.capsize (which controls quantile-derived asymmetric bars).
    
    # Phase 13.25.DF (Phase A): Quantile rendering style keys (AD-53)
    # These keys are INDEPENDENT — no cascading from profile.* keys.
    # Matches dfdraw's existing independent-keys pattern:
    # grid.alpha, scatter.alpha, hist.alpha, stats.alpha (style.py:37,41,48,73).
    "quantile.band.alpha": 0.25,
    # Alpha for quantile band rendering (fill_between). dfdraw design choice;
    # matplotlib's fill_between default is alpha=None (~1.0, fully opaque).
    "quantile.band.hatch": None,
    # Hatch pattern for quantile band (e.g., '//' for B&W printing). None = no hatch.
    "quantile.error_bars.capsize": 3.0,
    # Cap size for asymmetric error bars rendered when quantile_mode='error_bars'.
    # Independent of profile.capsize (which controls SEM/STD error bar caps in
    # non-quantile mode). Set both keys if you want consistent cap sizes across
    # rendering modes.
    "quantile.central_default": "mean",
    # Default central line when quantiles=[...] is set and central= is not
    # explicitly passed. 'mean' preserves backward compat with existing GB-mean
    # behavior per AD-45.
    
    # Auto-title (Phase 13.12.DF v1.2)
    "auto_title": False,
    "auto_title.fontsize": 10,
    "auto_title.sel_fontsize": 8,
    
    # Colors & markers
    "colors.palette": "tab10",
    "colors.categorical_palette": "tab20",
    "markers.cycle": ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h"],
    
    # Statistics box
    "stats.show": False,
    "stats.position": "upper right",
    "stats.fields": ["n", "mean", "std"],
    "stats.fontsize": 10,
    "stats.alpha": 0.8,
    "stats.boxstyle": "round",
    "stats.robust": False,  # Phase 13.6.G.DF: Use robust stats (median, MAD) for 1D
    
    # Legend
    "legend.outside": False,
    "legend.loc": "best",
    "legend.ncol": 1,
    "legend.frameon": True,
    
    # Group-by
    "groupby.mode": "overlay",
    "groupby.top_k": None,
    "groupby.other_label": "Other",
    
    # Sampling
    "sample.max_points": None,  # None = no limit
    "sample.random_state": 42,

    # ========================================================================
    # Phase 13.26.DF (Phase B): N-Channel Framework — channels.* namespace
    # ========================================================================
    # Per architect direction chat 2026-05-05: all assignment rules are style
    # parameters from day one. Defaults reproduce existing 1- and 2-channel
    # behaviour; 3-channel cases and quantile-touching paths are greenfield
    # (no production users — verified by grep on makeSmoothMapsWithTPC.py).
    #
    # See:
    #   - dfdraw/channels.py — Algorithm A implementation
    #   - docs/STYLING_FRAMEWORK_DECISIONS.md — AD-44 through AD-59
    #   - PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md §4

    # Priority lists used by Algorithm A Step 3 (greedy fallback for unknown
    # data-channel combinations). AD-55.
    "channels.priority.categorical": ["color", "linestyle", "marker"],
    "channels.priority.ordinal":     ["linestyle", "marker", "color"],

    # Visual-channel cycles. Replace drawer.py:527-528 module constants.
    # The discrete-quantile rendering path uses [1:] slicing on linestyle to
    # preserve FIX2's invariant that solid linestyle is reserved for the
    # central line (see channels.py and v1.2 §11.3).
    "channels.cycles.linestyle":   ["-", "--", "-.", ":"],
    "channels.cycles.marker":      ["o", "s", "^", "D", "v", "<", ">", "p"],
    "channels.cycles.color_count": 10,  # capacity for color overflow check

    # Per-data-channel pinned defaults. None = use Algorithm A explicit-case
    # rule from EXPLICIT_RULES; non-None = override the rule for this channel.
    # AD-56.
    "channels.default.vector":    None,
    "channels.default.group_by":  None,
    "channels.default.quantiles": None,

    # Overflow behaviour when a data channel's cardinality exceeds its
    # assigned visual channel's capacity. AD-58.
    #   "error" — raise ValueError with actionable suggestions (default)
    #   "warn"  — emit UserWarning and proceed with cycling
    "channels.overflow": "error",

    # Legend factoring. AD-59.
    #   True  — render one section per active data channel; entry count =
    #           sum of cardinalities (e.g., 5 + 3 + 3 = 11)
    #   False — flat deduplicated legend (existing FIX1 behaviour)
    "channels.legend.factored": True,

    # =========================================================================
    # Phase 13.28.DF: Robust data handling — AD-69 through AD-77
    # See docs/STYLING_FRAMEWORK_DECISIONS.md and proposal v1.1.
    # =========================================================================

    # NaN/inf handling policy. AD-70.
    #   "filter" — drop silently, counters reported in stats (default)
    #   "warn"   — drop + emit UserWarning with details
    #   "raise"  — raise ValueError on any NaN/inf
    "data.nan_policy": "filter",

    # Autorange strategy default. AD-73.
    # Determines range when range=None or range='auto'.
    # Options: "minmax", "hybrid" (default), "percentile_99",
    #          "percentile_95", "robust_3mad", "robust_4mad".
    "autorange.strategy": "hybrid",

    # Hybrid/robust autorange tunable parameters. AD-72, AD-76.
    # k_robust : MAD-sigma half-width for the robust window (used by
    #            'hybrid' and 'robust_*mad' strategies).
    # k_outlier: tolerance multiplier for outlier declaration on each
    #            side (used by 'hybrid' strategy only).
    "autorange.k_robust": 4.0,
    "autorange.k_outlier": 1.5,

    # Default percentile bounds for 'percentile_99' preset. AD-76.
    # Tuple form (low, high). 'percentile_95' bakes (2.5, 97.5).
    "autorange.percentile": (1, 99),
}

# =============================================================================
# Predefined Styles
# =============================================================================

PREDEFINED_STYLES: Dict[str, Dict[str, Any]] = {
    "default": {},  # Uses DEFAULT_STYLE as-is
    
    "publication": {
        "figure.figsize": (6, 4.5),
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": False,
        "scatter.size": 30,
        "scatter.alpha": 0.8,
        "hist.histtype": "step",
        "hist.linewidth": 1.5,
        "legend.frameon": False,
        "stats.show": False,
    },
    
    "presentation": {
        "figure.figsize": (10, 7),
        "figure.dpi": 100,
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "scatter.size": 80,
        "scatter.linewidths": 1.0,
        "axes.grid": True,
        "grid.alpha": 0.4,
    },
    
    "minimal": {
        "axes.grid": False,
        "legend.frameon": False,
        "scatter.edgecolors": "none",
        "hist.edgecolor": "none",
    },
}

# =============================================================================
# Global Style State
# =============================================================================

_current_style: Dict[str, Any] = copy.deepcopy(DEFAULT_STYLE)

# =============================================================================
# Style Functions
# =============================================================================

def get_style() -> Dict[str, Any]:
    """
    Get a copy of the current style dictionary.
    
    Returns
    -------
    dict
        Current style settings.
    """
    return copy.deepcopy(_current_style)


def set_style(style: Union[str, Dict[str, Any], None] = None) -> None:
    """
    Set the current drawing style.
    
    Parameters
    ----------
    style : str, dict, or None
        - str: Name of predefined style ("default", "publication", "presentation", "minimal")
        - dict: Custom style dictionary (merged with defaults)
        - None: Reset to default style
    
    Examples
    --------
    >>> set_style("publication")
    >>> set_style({"font.size": 14, "scatter.alpha": 0.5})
    >>> set_style(None)  # Reset to default
    """
    global _current_style
    
    if style is None:
        _current_style = copy.deepcopy(DEFAULT_STYLE)
        _apply_to_matplotlib()
        return
    
    if isinstance(style, str):
        if style not in PREDEFINED_STYLES:
            available = ", ".join(PREDEFINED_STYLES.keys())
            raise ValueError(f"Unknown style '{style}'. Available: {available}")
        _current_style = copy.deepcopy(DEFAULT_STYLE)
        _current_style.update(PREDEFINED_STYLES[style])
        _apply_to_matplotlib()
        return
    
    if isinstance(style, dict):
        # Validate keys
        invalid_keys = set(style.keys()) - set(DEFAULT_STYLE.keys())
        if invalid_keys:
            raise ValueError(f"Unknown style keys: {invalid_keys}")
        _current_style.update(style)
        _apply_to_matplotlib()
        return
    
    raise TypeError(f"style must be str, dict, or None, got {type(style)}")


def save_style(path: Union[str, Path]) -> None:
    """
    Save current style to JSON file.
    
    Parameters
    ----------
    path : str or Path
        Output file path (should end with .json).
    """
    path = Path(path)
    with open(path, 'w') as f:
        json.dump(_current_style, f, indent=2)


def load_style(path: Union[str, Path]) -> None:
    """
    Load style from JSON file and apply it.
    
    Parameters
    ----------
    path : str or Path
        Input file path.
    """
    path = Path(path)
    with open(path, 'r') as f:
        style = json.load(f)
    set_style(style)


def list_styles() -> list:
    """
    List available predefined style names.
    
    Returns
    -------
    list
        Names of predefined styles.
    """
    return list(PREDEFINED_STYLES.keys())


def _apply_to_matplotlib() -> None:
    """
    Apply current style settings to matplotlib rcParams where applicable.
    """
    # Map dfdraw style keys to matplotlib rcParams
    rcparams_map = {
        "figure.figsize": "figure.figsize",
        "figure.dpi": "figure.dpi",
        "figure.facecolor": "figure.facecolor",
        "font.size": "font.size",
        "axes.titlesize": "axes.titlesize",
        "axes.labelsize": "axes.labelsize",
        "xtick.labelsize": "xtick.labelsize",
        "ytick.labelsize": "ytick.labelsize",
        "legend.fontsize": "legend.fontsize",
        "axes.grid": "axes.grid",
        "grid.alpha": "grid.alpha",
        "grid.linestyle": "grid.linestyle",
    }
    
    for dfdraw_key, mpl_key in rcparams_map.items():
        if dfdraw_key in _current_style:
            plt.rcParams[mpl_key] = _current_style[dfdraw_key]


def get_style_value(key: str, default: Any = None) -> Any:
    """
    Get a single style value.
    
    Parameters
    ----------
    key : str
        Style key (e.g., "scatter.alpha").
    default : any
        Value to return if key not found.
    
    Returns
    -------
    any
        Style value.
    """
    return _current_style.get(key, default)
