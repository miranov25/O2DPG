"""
Auto-title helpers for dfdraw plot functions.

Phase 13.12.DF v1.2 — Automatic title generation from plot parameters.

Shared by: draw_profile, draw_hist, draw_hist2d, draw_hexbin.

Title format:
  Line 1 (main):  "y vs x  group:group_by  weights:w"
  Line 2 (sub):   "selection string" (italic, smaller font)

Usage in draw functions:
    from ._auto_title import build_auto_title, apply_auto_title, parse_auto_title_parts

    if title:
        ax.set_title(title)
    elif auto_title:
        parts = parse_auto_title_parts(auto_title)
        td = build_auto_title(x_name, y_name, group_by=group_by,
                              selection=selection, weights=weights, parts=parts)
        apply_auto_title(ax, td)
"""

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from ..style import get_style_value


def _text_width_points(text, fontsize, style="normal"):
    """Return rendered text width in typographic points.

    TextPath gives a backend-independent font-metric measurement, so title
    fitting depends on the actual figure width and font size rather than on a
    fixed character count.
    """
    if not text:
        return 0.0
    prop = FontProperties(size=fontsize, style=style)
    widths = []
    for line in str(text).splitlines() or [""]:
        if not line:
            widths.append(0.0)
            continue
        ext = TextPath((0, 0), line, prop=prop, usetex=False).get_extents()
        widths.append(float(ext.width) if np.isfinite(ext.width) else 0.0)
    return max(widths, default=0.0)


def fit_auto_title_text(text, fig, fontsize, width_fraction=0.94, style="italic"):
    """Fit one auto-title text line to the available figure width.

    The text is preserved verbatim when it fits.  Otherwise a longest-fitting
    prefix plus ``...`` is chosen using actual font metrics.  This replaces the
    historical fixed 72-character cut with a geometry-dependent policy.
    """
    if not text or fig is None:
        return text
    available = float(fig.get_figwidth()) * 72.0 * float(width_fraction)
    if available <= 0 or _text_width_points(text, fontsize, style=style) <= available:
        return text

    suffix = "..."
    if _text_width_points(suffix, fontsize, style=style) > available:
        return suffix

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + suffix
        if _text_width_points(candidate, fontsize, style=style) <= available:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + suffix


def fit_auto_title_to_figure(fig, title_dict, sub_fontsize=None):
    """Return a title dict whose selection subtitle fits ``fig`` width."""
    if sub_fontsize is None:
        sub_fontsize = get_style_value("auto_title.sel_fontsize", 8)
    fitted = dict(title_dict)
    if fitted.get("sub"):
        fitted["sub"] = fit_auto_title_text(
            fitted["sub"], fig, sub_fontsize, style="italic"
        )
    return fitted


def parse_auto_title_parts(auto_title):
    """Parse auto_title parameter into set of parts.

    Parameters
    ----------
    auto_title : bool or str
        True / "all" → {"expr", "group", "weights", "sel"}
        "expr"       → {"expr"}
        "expr+group" → {"expr", "group"}

    Returns
    -------
    set of str
    """
    if auto_title is True or auto_title == "all":
        return {"expr", "group", "weights", "sel"}
    if isinstance(auto_title, str):
        return set(auto_title.split("+"))
    return set()


def format_profile_quantile_semantics(quantiles):
    """Concise, deterministic non-default profile-quantile metadata.

    Two values denote a pair; three or more are listed explicitly to avoid
    implying that every set of quantiles is one continuous interval.
    """
    if quantiles is None or len(quantiles) == 0:
        return None
    vals = [f"{float(q):.6g}" for q in quantiles]
    if len(vals) == 2:
        return f"q {vals[0]}–{vals[1]}"
    return "q [" + ", ".join(vals) + "]"


def build_auto_title(x, y=None, group_by=None, selection=None,
                     weights=None, parts=("expr", "group", "sel"),
                     quantiles=None):
    """Build automatic title dict from plot parameters.

    Returns dict with 'main' (str) and 'sub' (str or None).
    Separate strings so matplotlib can apply different font sizes.

    Parameters
    ----------
    x : str
        X-axis expression name.
    y : str or None
        Y-axis expression name (None for 1D histograms).
    group_by : str or None
        Grouping variable name.
    selection : str or None
        Selection string. Non-string selections (callable, array) are
        silently skipped per review P1.2.
    weights : str or None
        Weight column name.
    parts : set or tuple
        Which parts to include: "expr", "group", "weights", "sel".

    Returns
    -------
    dict
        {"main": str, "sub": str or None}
    """
    parts = set(parts)

    # Line 1: expression + group + weights
    main = ""
    if "expr" in parts:
        main = f"{y} vs {x}" if y else str(x)
    if "group" in parts and group_by:
        main += f"  group:{group_by}"
    if "weights" in parts and weights:
        main += f"  weights:{weights}"

    # Line 2: selection (only if string, per review P1.2)
    sub = None
    if "sel" in parts and isinstance(selection, str) and selection:
        # Preserve the semantic selection string here.  Width-dependent fitting
        # belongs to the rendering target, where the figure geometry is known.
        sub = selection

    q_semantics = format_profile_quantile_semantics(quantiles)
    if q_semantics:
        sub = f"{sub}; {q_semantics}" if sub else q_semantics
    return {"main": main, "sub": sub}


def apply_auto_title(ax, title_dict, fontsize=None, sub_fontsize=None):
    """Apply auto-title dict to axes with two font sizes.

    Parameters
    ----------
    ax : matplotlib Axes
    title_dict : dict with 'main' and 'sub' keys
    fontsize : int or None
        Main title font size. Default from style or 10.
    sub_fontsize : int or None
        Selection subtitle font size. Default from style or 8.
    """
    if fontsize is None:
        fontsize = get_style_value("auto_title.fontsize", 10)
    if sub_fontsize is None:
        sub_fontsize = get_style_value("auto_title.sel_fontsize", 8)

    fitted = fit_auto_title_to_figure(ax.figure, title_dict, sub_fontsize=sub_fontsize)
    has_sub = bool(fitted.get("sub"))
    if has_sub:
        # Subtitle between plot frame and main title
        sub_artist = ax.text(0.5, 1.01, fitted["sub"],
                transform=ax.transAxes, fontsize=sub_fontsize,
                ha='center', va='bottom', style='italic', color='0.4')
        # Phase 13.13.DF: Mark for identification by append_auto_title
        sub_artist._is_auto_subtitle = True
        # Keep the unabridged semantic text for later same=True merging.
        ax._auto_title_subs = [title_dict["sub"]]
    if title_dict.get("main"):
        # Push main title up when subtitle present to avoid overlap
        pad = 20 if has_sub else 6
        ax.set_title(title_dict["main"], fontsize=fontsize, pad=pad)


def append_auto_title(ax, title_dict, fontsize=None, sub_fontsize=None):
    """Append new expression to existing title for same=True overlay.

    AD-18: Append to title when same=True + auto_title=True.
    AD-35: No hard limit on title lines.

    Parameters
    ----------
    ax : matplotlib Axes
    title_dict : dict with 'main' and 'sub' keys
    fontsize : int or None
        Main title font size. Default from style or 10.
    sub_fontsize : int or None
        Selection subtitle font size. Default from style or 8.
    """
    if fontsize is None:
        fontsize = get_style_value("auto_title.fontsize", 10)
    if sub_fontsize is None:
        sub_fontsize = get_style_value("auto_title.sel_fontsize", 8)

    existing = ax.get_title()

    if existing and title_dict.get("main"):
        # Append main line to existing title
        new_title = existing.rstrip() + "\n" + title_dict["main"]
        ax.set_title(new_title, fontsize=fontsize)
    elif title_dict.get("main"):
        # No existing title — apply as first title
        ax.set_title(title_dict["main"], fontsize=fontsize)

    # Handle subtitle merging (stored in axes._auto_title_subs)
    if title_dict.get("sub"):
        if not hasattr(ax, '_auto_title_subs'):
            ax._auto_title_subs = []
        ax._auto_title_subs.append(title_dict["sub"])
        # Merge the full semantic strings, then fit only the displayed text.
        merged = "; ".join(ax._auto_title_subs)
        merged = fit_auto_title_text(
            merged, ax.figure, sub_fontsize, style="italic"
        )
        # Remove old subtitle texts and redraw
        for txt in list(ax.texts):
            if getattr(txt, '_is_auto_subtitle', False):
                txt.remove()
        sub_artist = ax.text(
            0.5, 1.01, merged,
            transform=ax.transAxes, fontsize=sub_fontsize,
            ha='center', va='bottom', style='italic', color='0.4')
        sub_artist._is_auto_subtitle = True
        # Ensure title has padding for subtitle
        current_title = ax.get_title()
        if current_title:
            ax.set_title(current_title, fontsize=fontsize, pad=20)


def resolve_auto_title(auto_title):
    """Resolve auto_title: per-call value > style default.

    If auto_title is False (not set by caller), check style.
    """
    if auto_title is False:
        return get_style_value("auto_title", False)
    return auto_title
