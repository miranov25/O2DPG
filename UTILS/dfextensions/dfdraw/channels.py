"""dfdraw.channels — N-Channel Framework (Phase 13.26.DF Phase B)

Implements Algorithm A from brainstorm v1.4 §3.1: automatic visual-channel
assignment for N data channels (vector, group_by, quantiles, selection_delta,
...).

Per Phase 13.26.DF v1.2 §5.2, the API is N-channel from day one:

    channels: list[DataChannel] -> dict[str, str]

Phase B populates ``[vector, group_by, quantiles]``. Phase D appends
``DataChannel('selection_delta', ...)``. Future phases extend by appending more
``DataChannel`` entries — no signature change is ever required to add a data
channel type.

References
----------
- Phase 13.26.DF v1.2 Proposal — N-Channel Framework
- Brainstorm v1.4 §3.1 (Algorithm A), §3.2 (default-style table), §10 (style
  sets), §13 (default-style table deliverable), §14 (phasing plan)
- AD-44 through AD-59 — see ``docs/STYLING_FRAMEWORK_DECISIONS.md``
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

from .style import get_style_value


# ----------------------------------------------------------------------------
# Data channel record (G-7: forward-extensibility — list-based API)
# ----------------------------------------------------------------------------

@dataclass
class DataChannel:
    """One source of multiplicity in a draw call.

    A "data channel" is anything that produces N >= 2 graphs from a single
    user call. dfdraw recognises three in Phase B:

    - ``'vector'``      — vector expressions ``[A, B, C]:X`` produce one graph
                           per element
    - ``'group_by'``    — grouping by a column produces one graph per group
    - ``'quantiles'``   — discrete-mode quantile rendering produces one graph
                           per quantile value

    Phase D will add ``'selection_delta'``. Future phases may add more.

    Channels with ``cost == 0`` (zero-cost rendering modes: ``'band'``,
    ``'error_bars'``, ``'nested_band'``) consume no visual channel slot in
    Algorithm A — they ride on top of the central line or as fill regions.

    Parameters
    ----------
    name : str
        Channel identifier. Must match the keys used in ``EXPLICIT_RULES``,
        the ``channels.default.<name>`` style namespace, and the per-call
        kwarg ``<name>_style``.
    is_categorical : bool
        ``True`` for categorical channels (vector, group_by); ``False`` for
        ordinal channels (quantiles, selection_delta). Determines which
        priority list (``channels.priority.categorical`` vs
        ``channels.priority.ordinal``) the greedy fallback consults.
    cardinality : int
        Number of distinct values. Used by Algorithm A's capacity check
        (Step 5) and by the explicit-case rule footnote on the 3-channel
        marker assignment.
    requested_style : Optional[str], default None
        Per-call kwarg override (e.g., ``vector_style='marker'``). Highest
        precedence in the resolution chain.
    cost : int, default 1
        Visual-channel slots consumed. ``0`` for zero-cost modes; ``1`` for
        one-channel modes (the typical case).
    """

    name: str
    is_categorical: bool
    cardinality: int
    requested_style: Optional[str] = None
    cost: int = 1


# ----------------------------------------------------------------------------
# Explicit-case rules (Algorithm A Step 2)
# ----------------------------------------------------------------------------
#
# Keyed on frozenset of channel names. Phase D adds entries for combinations
# involving 'selection_delta'. Per Coder Quick Reference Card directive
# (v1.2 §11.2 directive 4): future phases append entries — never modify
# existing ones (additive contract).
#
# Source: Phase 13.26.DF v1.2 §3.2 default-style table.
# Provenance: AD-56 (3-channel default), brainstorm v1.4 §13 deliverable table.

EXPLICIT_RULES: "dict[frozenset, dict[str, str]]" = {
    # 1-channel cases
    frozenset({'vector'}):    {'vector':    'color'},
    frozenset({'group_by'}):  {'group_by':  'color'},
    frozenset({'quantiles'}): {'quantiles': 'linestyle'},
    # 2-channel cases
    frozenset({'vector', 'group_by'}):
        {'group_by': 'color', 'vector': 'linestyle'},
    frozenset({'group_by', 'quantiles'}):
        {'group_by': 'color', 'quantiles': 'linestyle'},
    frozenset({'vector', 'quantiles'}):
        {'vector': 'color', 'quantiles': 'linestyle'},
    # 3-channel case (AD-56)
    frozenset({'vector', 'group_by', 'quantiles'}):
        {'group_by': 'color', 'vector': 'marker', 'quantiles': 'linestyle'},
    # Phase D: append 'selection_delta' combinations here.
}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def assign_channels(channels: Sequence[DataChannel]) -> "dict[str, str]":
    """Algorithm A: assign visual channels to active data channels.

    Resolution order per channel (per v1.2 §5.2):

    1. ``channel.requested_style`` — per-call kwarg, always wins
    2. Style pinned default ``channels.default.<name>``
    3. Explicit-case rule ``EXPLICIT_RULES[frozenset({c.name for c in active})]``
    4. Greedy priority-list fallback (``channels.priority.categorical`` /
       ``channels.priority.ordinal``)

    Then:

    5. Collision check (Step 4): no two channels assigned to the same visual
    6. Capacity check (Step 5): cardinality must not exceed cycle capacity

    Parameters
    ----------
    channels : Sequence[DataChannel]
        All data channels for this draw call. Zero-cost channels (cost=0)
        are tracked in the input list but absent from the output dict.

    Returns
    -------
    dict[str, str]
        Mapping ``{channel_name: visual_channel}`` for channels with cost > 0.

    Raises
    ------
    ValueError
        On channel collision (Step 4) or capacity overflow (Step 5) when
        ``channels.overflow`` style key is ``"error"`` (default).
    """
    # Step 0: filter to active channels with cost > 0 (zero-cost modes
    # consume no visual channel slot per brainstorm v1.4 §3.1 Step 0).
    active = [c for c in channels if c.cost > 0]

    if not active:
        return {}

    result: "dict[str, str]" = {}
    used_visual: "set[str]" = set()

    # Priority 1: per-call kwarg (highest precedence — always wins).
    # Priority 2: style pinned default `channels.default.<name>`.
    for c in active:
        if c.requested_style is not None:
            result[c.name] = c.requested_style
            used_visual.add(c.requested_style)
        else:
            pinned = get_style_value(f"channels.default.{c.name}")
            if pinned is not None:
                result[c.name] = pinned
                used_visual.add(pinned)

    # Priority 3: explicit-case rule for known patterns (G-7: keyed on
    # frozenset of all active-channel names; entries that conflict with
    # already-resolved channels are skipped, leaving them for Priority 4).
    unresolved_names = [c.name for c in active if c.name not in result]
    if unresolved_names:
        active_set = frozenset(c.name for c in active)
        rule = EXPLICIT_RULES.get(active_set)
        if rule is not None:
            still_unresolved = []
            for name in unresolved_names:
                visual = rule.get(name)
                if visual is not None and visual not in used_visual:
                    result[name] = visual
                    used_visual.add(visual)
                else:
                    still_unresolved.append(name)
            unresolved_names = still_unresolved

    # Priority 4: greedy priority-list fallback (Step 3 in §3.1).
    # Used when (a) the active-set isn't in EXPLICIT_RULES (e.g., a future
    # combination not yet covered) or (b) per-call kwargs displaced the
    # explicit rule's intended assignment.
    if unresolved_names:
        cat_priority = get_style_value(
            "channels.priority.categorical",
            ["color", "linestyle", "marker"],
        )
        ord_priority = get_style_value(
            "channels.priority.ordinal",
            ["linestyle", "marker", "color"],
        )
        active_by_name = {c.name: c for c in active}
        # Sort: categorical first, then by descending cardinality
        # (per brainstorm v1.4 §3.1 Step 1).
        unresolved_sorted = sorted(
            unresolved_names,
            key=lambda n: (
                0 if active_by_name[n].is_categorical else 1,
                -active_by_name[n].cardinality,
            ),
        )
        overflow_mode = get_style_value("channels.overflow", "error")
        for name in unresolved_sorted:
            c = active_by_name[name]
            priority = cat_priority if c.is_categorical else ord_priority
            assigned = None
            for visual in priority:
                if visual not in used_visual:
                    assigned = visual
                    break
            if assigned is None:
                msg = (
                    f"Cannot assign visual channel for '{name}' "
                    f"({c.cardinality} values). "
                    f"All channels in use: {sorted(used_visual)}. "
                    f"Reduce active dimensions or use facet=True."
                )
                if overflow_mode == "error":
                    raise ValueError(msg)
                warnings.warn(msg, UserWarning, stacklevel=2)
                assigned = priority[0]  # graceful fallback under "warn"
            result[name] = assigned
            used_visual.add(assigned)

    # Step 4: collision check. Two channels assigned to the same visual is
    # an error — typically caused by per-call kwargs or pinned defaults that
    # conflict with the explicit rule.
    inverse: "dict[str, str]" = {}
    for name, visual in result.items():
        if visual in inverse:
            other = inverse[visual]
            raise ValueError(
                f"Channel collision: '{name}' and '{other}' both assigned "
                f"to '{visual}'. "
                f"Override with {name}_style= or {other}_style=, "
                f"or change channels.default.* in style."
            )
        inverse[visual] = name

    # Step 5: capacity check. A data channel's cardinality must fit within
    # its assigned visual channel's cycle length.
    cycle_lengths = {
        'color': get_style_value("channels.cycles.color_count", 10),
        'linestyle': len(get_style_value(
            "channels.cycles.linestyle", ["-", "--", "-.", ":"])),
        'marker': len(get_style_value(
            "channels.cycles.marker",
            ["o", "s", "^", "D", "v", "<", ">", "p"])),
    }
    overflow_mode = get_style_value("channels.overflow", "error")
    for c in active:
        visual = result.get(c.name)
        if visual is None:
            continue
        capacity = cycle_lengths.get(visual, 10)
        if c.cardinality > capacity:
            msg = (
                f"{c.name} ({c.cardinality} values) exceeds "
                f"{visual} cycle capacity ({capacity}).\n"
                f"  Options: top_k={capacity}, "
                f"facet=True, group_by_bins={capacity}"
            )
            if overflow_mode == "error":
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)

    return result


def build_factored_legend(ax, assignment, channel_entries, **legend_kwargs):
    """Build a factored legend with one section per data channel.

    Per v1.2 §6: when ``channels.legend.factored`` is True (default) and
    >= 2 channels are active, render section headers per channel with
    proxy artists for each entry. Total entries = sum(cardinalities) not
    product (e.g., 5 + 3 + 3 = 11 instead of 5 * 3 * 3 = 75).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    assignment : dict[str, str]
        Output of ``assign_channels()`` — maps channel name to visual channel.
    channel_entries : dict[str, list[tuple[str, Any]]]
        Per-channel entries to render::

            {
                'vector':    [('dy_I0', 'o'), ('dy_I1', 's'), ...],
                'group_by':  [('mP4 in [0,2)', 'C0'), ...],
                'quantiles': [('q=10%', '--'), ...],
            }

        Each tuple is (label, visual-style-value).
    **legend_kwargs
        Forwarded to ``ax.legend()`` (e.g., ``loc``, ``fontsize``).

    Returns
    -------
    matplotlib.legend.Legend or None
        Returns None if no entries to render.
    """
    import matplotlib.pyplot as plt

    handles = []
    labels = []

    # Stable section order: keep the order in which channels appear in
    # the assignment dict (Python 3.7+ guarantees insertion order).
    for ch_name, visual in assignment.items():
        entries = channel_entries.get(ch_name, [])
        if not entries:
            continue
        # Section header — invisible artist with a section text label.
        handles.append(plt.Line2D([], [], color='none', label=''))
        labels.append(f"— {ch_name} ({visual}) —")
        # Per-entry proxy artist matched to the visual channel.
        for entry_label, style_value in entries:
            if visual == 'color':
                h = plt.Line2D(
                    [], [], color=style_value, marker='s',
                    linestyle='none', markersize=8,
                )
            elif visual == 'linestyle':
                h = plt.Line2D(
                    [], [], color='gray', linestyle=style_value,
                    linewidth=1.5,
                )
            elif visual == 'marker':
                h = plt.Line2D(
                    [], [], color='gray', marker=style_value,
                    linestyle='none', markersize=8,
                )
            else:
                # Unknown visual — neutral fallback.
                h = plt.Line2D([], [], color='gray')
            handles.append(h)
            labels.append(entry_label)

    if not handles:
        return None

    legend_defaults = {
        'loc': get_style_value("legend.loc", "best"),
        'frameon': get_style_value("legend.frameon", True),
    }
    legend_defaults.update(legend_kwargs)
    return ax.legend(handles, labels, **legend_defaults)


__all__ = [
    'DataChannel',
    'EXPLICIT_RULES',
    'assign_channels',
    'build_factored_legend',
]
