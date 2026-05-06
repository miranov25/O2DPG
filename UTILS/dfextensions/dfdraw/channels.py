"""dfdraw.channels — N-Channel Framework (Phase 13.26.DF Phase B)

This module implements Algorithm A from brainstorm v1.4 §3.1: automatic
visual-channel assignment for N data channels (vector, group_by, quantiles,
selection_delta, ...).

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

Status
------
Phase 13.26.DF Commit 1 (scaffolding): module + dataclass + EXPLICIT_RULES
table populated with the 7 Phase B entries. Function bodies raise
``NotImplementedError("Phase 13.26.DF Commit 2: implementation pending")`` so
that callers wired up in Commit 2 fail loudly until the algorithm lands.

Existing 578 tests do not call this module (Commit 1 is pure scaffolding).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


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

    Examples
    --------
    Phase B 3-channel call::

        channels = [
            DataChannel('vector',    is_categorical=True,  cardinality=3),
            DataChannel('group_by',  is_categorical=True,  cardinality=5),
            DataChannel('quantiles', is_categorical=False, cardinality=3,
                        cost=1),
        ]
        # -> assign_channels(channels) returns:
        #   {'group_by': 'color', 'vector': 'marker', 'quantiles': 'linestyle'}

    Phase D extensibility (no API change)::

        channels.append(DataChannel('selection_delta',
                                    is_categorical=False,
                                    cardinality=4, cost=0))
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

EXPLICIT_RULES: dict[frozenset[str], dict[str, str]] = {
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
# Public API (stubs — Commit 2 implements)
# ----------------------------------------------------------------------------

def assign_channels(channels: list[DataChannel]) -> dict[str, str]:
    """Algorithm A: assign visual channels to active data channels.

    Resolution order per channel (per v1.2 §5.2):

    1. ``channel.requested_style`` — per-call kwarg, always wins
    2. Style pinned default ``channels.default.<name>``
    3. Explicit-case rule ``EXPLICIT_RULES[frozenset({c.name for c in active})]``
    4. Greedy priority-list fallback (``channels.priority.categorical`` /
       ``channels.priority.ordinal``)

    Parameters
    ----------
    channels : list[DataChannel]
        All data channels for this draw call. Zero-cost channels (cost=0)
        are tracked in the input list but absent from the output dict.

    Returns
    -------
    dict[str, str]
        Mapping ``{channel_name: visual_channel}`` for channels with cost > 0.

    Raises
    ------
    ValueError
        On channel collision (Step 4) or capacity overflow (Step 5).
        Behaviour controlled by ``channels.overflow`` style key.

    Notes
    -----
    Idempotency contract per v1.2 §5.5: this function is called at most once
    per top-level user call. The vector path (``_draw_vector()``) calls it
    once and forwards the resolved styles via kwargs to ``draw_profile()``;
    ``draw_profile()`` only calls ``assign_channels()`` itself when not
    invoked from a vector parent (detected via ``quantile_style is None``
    at entry).
    """
    raise NotImplementedError(
        "Phase 13.26.DF Commit 2: implementation pending. "
        "Scaffolding only — see PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md §5.2."
    )


def build_factored_legend(ax, assignment: dict[str, str], channel_entries: dict):
    """Build a factored legend with one section per data channel.

    Per v1.2 §6: when ``channels.legend.factored`` is True (default) and
    >= 2 channels are active, render section headers per channel with
    proxy artists for each entry. Total entries = sum(cardinalities) not
    product (e.g., 5 + 3 + 3 = 11 instead of 5 * 3 * 3 = 75).

    When ``channels.legend.factored`` is False, fall through to the existing
    flat deduplicated legend (``_add_vector_main_legend_dedup`` from FIX1).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes to attach the legend to.
    assignment : dict[str, str]
        Output of ``assign_channels()`` — maps channel name to visual channel.
    channel_entries : dict
        Per-channel entries to render in the legend, e.g.::

            {
                'vector':    [('dy_I0', 'o'), ('dy_I1', 's'), ...],
                'group_by':  [('mP4 in [0,2)', 'C0'), ...],
                'quantiles': [('q=10%', '--'), ...],
            }

    Returns
    -------
    matplotlib.legend.Legend
        The factored legend artist attached to ``ax``.
    """
    raise NotImplementedError(
        "Phase 13.26.DF Commit 2: implementation pending. "
        "Scaffolding only — see PHASE_13_26_DF_v1_2_Proposal_NChannelFramework.md §6."
    )


__all__ = [
    'DataChannel',
    'EXPLICIT_RULES',
    'assign_channels',
    'build_factored_legend',
]
