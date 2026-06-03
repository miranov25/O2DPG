"""Phase 13.50.DF — Legend polymorphism normalizer + applier.

Provides the third polymorphism normalizer in dfdraw (after ``fit=`` and
``summary_fit=``): accepts ``bool | str | dict | None`` and produces a
canonical dict for ``_apply_legend_mode`` to consume.

Both ``legend=`` (polymorphic) and ``show_legend=`` (bool simple-form) are
new parallel public kwargs introduced in Phase 13.50 step 4. The framing
in the v2.5 proposal calls show_legend= "back-compat", but R16 source-truth
shows it never existed at the public dispatcher level before this phase —
it's a NEW additive kwarg, same as ``legend=``. Both reach this normalizer.

Modes — once normalized:
  - 'all'    — per-panel legends drawn (existing behavior; matplotlib default)
  - 'none'   — strip every per-axes legend; no fig-level legend either
  - 'shared' — capture handles/labels from the first axes that has a legend,
               strip per-axes legends, draw a single ``fig.legend(...)``
  - 'first'  — keep ``fig.axes[0]``'s legend; strip all others

A canonical dict of ``None`` means "no override given; preserve default
behavior" — both normalizer and applier no-op for None. This keeps
back-compat for every existing call that doesn't pass either kwarg.

Source-verified at PHASE_13_49_DF_FIX1_END (a6ddc753) before write per
Coder QRC v1.29 R16.
"""

from typing import Any, Dict, Optional, Union


# --------------------------------------------------------------------------- #
# Validation constants — accepted modes + accepted scalar shortcuts
# --------------------------------------------------------------------------- #

_VALID_MODES = frozenset({'all', 'none', 'shared', 'first'})

# Allowed keys in a user-supplied legend= dict; unknown keys raise.
# Mirrors matplotlib's Legend kwargs that make sense at our abstraction level.
_ALLOWED_DICT_KEYS = frozenset({
    'mode', 'loc', 'ncol', 'bbox_to_anchor', 'frameon', 'fontsize', 'title',
})


# --------------------------------------------------------------------------- #
# Normalizer
# --------------------------------------------------------------------------- #

def _normalize_legend_spec(legend: Optional[Union[bool, str, Dict[str, Any]]],
                           show_legend: Optional[bool] = None
                           ) -> Optional[Dict[str, Any]]:
    """Map (``legend``, ``show_legend``) inputs to a canonical dict (or None).

    Precedence
    ----------
    1. ``legend=`` wins if both are passed (no warning emitted; the two kwargs
       are parallel, not deprecation-aliased).
    2. ``show_legend=`` maps: True → ``{'mode': 'all'}``, False → ``{'mode': 'none'}``.
    3. Both None → returns None (preserve current default; applier no-ops).

    Polymorphic forms of ``legend=`` (shortcut → canonical)
    -------------------------------------------------------
    - ``True``           → ``{'mode': 'all'}``
    - ``False``          → ``{'mode': 'none'}``
    - ``'all'/'none'/'shared'/'first'`` → ``{'mode': <that>}``
    - ``dict``           → merged with defaults; 'mode' key required

    Returns
    -------
    dict | None
        Canonical legend spec with keys:
          ``{'mode': str, 'loc': str|None, 'ncol': int|None,
             'bbox_to_anchor': tuple|None, 'frameon': bool,
             'fontsize': str|int|None, 'title': str|None}``
        or ``None`` to mean "no override; preserve default behavior".

    Raises
    ------
    ValueError
        On invalid mode string, invalid dict shape, unknown dict key, or
        wrong type for ``legend`` / ``show_legend``.

    Idempotency
    -----------
    ``_normalize_legend_spec(_normalize_legend_spec(x)) ==
    _normalize_legend_spec(x)`` for every accepted input form, including the
    show_legend → legend mapping path. Locked by
    ``test_normalize_legend_spec_idempotent`` (invariance layer).
    """
    # No override on either path → no-op signal to the applier.
    if legend is None and show_legend is None:
        return None

    # legend= wins if both passed.
    if legend is None:
        if not isinstance(show_legend, bool):
            raise ValueError(
                f"show_legend must be bool or None, got {type(show_legend).__name__}"
            )
        return _build_canonical(mode='all' if show_legend else 'none')

    # legend= is set; show_legend= (if any) is ignored.
    if isinstance(legend, bool):
        return _build_canonical(mode='all' if legend else 'none')

    if isinstance(legend, str):
        if legend not in _VALID_MODES:
            raise ValueError(
                f"legend mode must be one of {sorted(_VALID_MODES)}, "
                f"got {legend!r}"
            )
        return _build_canonical(mode=legend)

    if isinstance(legend, dict):
        unknown = set(legend) - _ALLOWED_DICT_KEYS
        if unknown:
            raise ValueError(
                f"legend dict has unknown keys {sorted(unknown)}; "
                f"allowed keys are {sorted(_ALLOWED_DICT_KEYS)}"
            )
        mode = legend.get('mode')
        if mode is None or mode not in _VALID_MODES:
            raise ValueError(
                f"legend dict must include 'mode' key with value in "
                f"{sorted(_VALID_MODES)}, got mode={mode!r}"
            )
        return _build_canonical(**legend)

    raise ValueError(
        f"legend must be bool | str | dict | None, got {type(legend).__name__}"
    )


def _build_canonical(*, mode: str,
                     loc: Optional[str] = None,
                     ncol: Optional[int] = None,
                     bbox_to_anchor: Optional[tuple] = None,
                     frameon: bool = True,
                     fontsize: Optional[Union[str, int]] = None,
                     title: Optional[str] = None) -> Dict[str, Any]:
    """Build a canonical legend dict with all keys present (None for absent)."""
    return {
        'mode':           mode,
        'loc':            loc,
        'ncol':           ncol,
        'bbox_to_anchor': bbox_to_anchor,
        'frameon':        frameon,
        'fontsize':       fontsize,
        'title':          title,
    }


# --------------------------------------------------------------------------- #
# Applier — manipulates the figure to match the canonical spec
# --------------------------------------------------------------------------- #

def _apply_legend_mode(fig, canonical: Optional[Dict[str, Any]]) -> None:
    """Apply the canonical legend spec to ``fig`` in place.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    canonical : dict or None
        The output of ``_normalize_legend_spec``. ``None`` is a no-op
        (preserve current default behavior — per-axes legends drawn by
        the existing dispatcher remain untouched).

    Modes
    -----
    - 'all'    : no-op (per-axes legends already drawn).
    - 'none'   : strip every per-axes legend; no fig-level added.
    - 'first'  : keep ``fig.axes[0]``'s legend; strip the rest.
    - 'shared' : capture handles+labels from the first axes that has a legend,
                 strip all per-axes legends, draw a single ``fig.legend(...)``
                 with the canonical loc / ncol / bbox_to_anchor / frameon /
                 fontsize / title (each optional).
    """
    if canonical is None:
        return  # no override — keep current default

    mode = canonical['mode']

    if mode == 'all':
        return  # no-op

    if mode == 'none':
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is not None:
                leg.remove()
        return

    if mode == 'first':
        first_seen = False
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            if not first_seen:
                first_seen = True
                continue  # keep this one
            leg.remove()
        return

    if mode == 'shared':
        # Capture handles+labels from the first axes that has a legend;
        # then strip per-axes legends and add a single fig-level one.
        handles, labels = None, None
        for ax in fig.axes:
            leg = ax.get_legend()
            if leg is None:
                continue
            if handles is None:
                # Use the axes' get_legend_handles_labels (not the legend's
                # internal artists — those are proxies that don't replay well
                # at fig level on some matplotlib versions).
                handles, labels = ax.get_legend_handles_labels()
            leg.remove()
        if handles:
            # Compose kwargs from canonical, dropping None values so matplotlib
            # uses its own defaults where the user didn't specify.
            fig_legend_kwargs = {
                k: v for k, v in {
                    'loc':            canonical['loc'] or 'lower center',
                    'ncol':           canonical['ncol'] or 1,
                    'bbox_to_anchor': canonical['bbox_to_anchor'],
                    'frameon':        canonical['frameon'],
                    'fontsize':       canonical['fontsize'],
                    'title':          canonical['title'],
                }.items() if v is not None
            }
            fig.legend(handles, labels, **fig_legend_kwargs)
        return

    # Defensive: _normalize_legend_spec should have rejected anything else.
    raise ValueError(f"_apply_legend_mode: unknown mode {mode!r}")
