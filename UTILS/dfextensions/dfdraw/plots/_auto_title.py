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
from ..style import get_style_value

_AUTO_TITLE_MAX_SEL_LEN = 72  # Truncate selection string beyond this


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


def build_auto_title(x, y=None, group_by=None, selection=None,
                     weights=None, parts=("expr", "group", "sel")):
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
        sub = selection
        if len(sub) > _AUTO_TITLE_MAX_SEL_LEN:
            sub = sub[:_AUTO_TITLE_MAX_SEL_LEN - 3] + "..."

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

    has_sub = bool(title_dict.get("sub"))
    if has_sub:
        # Subtitle between plot frame and main title
        sub_artist = ax.text(0.5, 1.01, title_dict["sub"],
                transform=ax.transAxes, fontsize=sub_fontsize,
                ha='center', va='bottom', style='italic', color='0.4')
        # Phase 13.13.DF: Mark for identification by append_auto_title
        sub_artist._is_auto_subtitle = True
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
        # Update merged subtitle text
        merged = "; ".join(ax._auto_title_subs)
        if len(merged) > _AUTO_TITLE_MAX_SEL_LEN:
            merged = merged[:_AUTO_TITLE_MAX_SEL_LEN - 3] + "..."
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
