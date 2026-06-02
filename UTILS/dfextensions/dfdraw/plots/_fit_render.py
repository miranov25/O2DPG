"""Phase 13.42.DF — Inline-fit overlay & textbox rendering.

Pure-presentation helpers consumed by ``_apply_fits`` in each plot module.
Produces:
  - ``ax.lines`` entries: one per overlay; ``fit.linestyle_cycle`` when
    multiple fits stack on the same curve.
  - ``ax.texts`` entries: one textbox per ax, per-fit blocks for any fit
    with ``show_params`` not False.

Position policy (Phase 13.42 v1.3 CP2-2):
  fit textbox at ``style.fit.position`` (default 'upper left')
  stats textbox at ``style.stats.position`` (default 'upper right')
  matplotlib legend at ``legend.loc='best'`` (often 'upper right')
  → the three never collide by default.
"""

from typing import List, Dict, Any, Optional, Tuple

import numpy as np

# Style key lookup with safe fallback (the style dict is patched into
# DEFAULT_STYLE in style.py; we read via get_style_value() so set_style()
# updates take effect at draw time, not at module-import time).
# Phase 13.42.DF FIX1 (B1 deeper root cause): v1.0 used `get_style(key)` but
# `get_style()` takes no args — TypeError was caught and the default was
# returned, making set_style({'fit.text_fontsize_facet': N}) silently no-op.
def _style_get(key: str, default):
    try:
        from ..style import get_style_value
        return get_style_value(key, default)
    except Exception:
        try:
            from dfdraw.style import get_style_value
            return get_style_value(key, default)
        except Exception:
            return default


# ---------------------------------------------------------------------------
# Phase 13.50.DF — Display-name map (render-only)
# ---------------------------------------------------------------------------
# Canonical Python identifiers in plots/fits.py (slope, intercept, amplitude,
# center, sigma, decay) are UNCHANGED. This map is consulted ONLY at render
# time to produce shorter / mathtext display names in fit textboxes and
# summary_fit tables. The _valid_fields whitelist (below at ~line 190) still
# accepts canonical names, so users pass show_fields=['slope','chi2'] etc. by
# canonical name; the renderer alone does the display swap.
#
# Rationale for p0/p1: _polynomial_factory in plots/fits.py builds ascending
# powers (c0 + c1*x + c2*x^2 + ...), so c0=constant=intercept and
# c1=x-coefficient=slope. The linear mapping follows the same convention:
# intercept → p0 (constant), slope → p1 (x-coefficient).
_DISPLAY_NAMES = {
    'slope':     'p1',
    'intercept': 'p0',
    'amplitude': 'A',
    'center':    r'$\mu$',
    'sigma':     r'$\sigma$',
    'decay':     r'$\tau$',
}


def _resolve_display_name(canonical, rename_params=None):
    """Map a canonical param name to its display form.

    Resolution order (highest priority first):
      1. ``rename_params={old: new}`` — per-call override from
         ``fit_textbox_kwargs`` (wired in Phase 13.50 step 3; for step 1 the
         arg is always None and the branch is unused).
      2. ``_DISPLAY_NAMES`` — module-level short/Greek-mathtext map.
      3. The literal canonical name — for callable-fit param names or any
         identifier not in the map.

    Parameters
    ----------
    canonical : str
        The canonical Python identifier (as supplied by ``param_names`` from
        ``dispatch_fit`` results — same names as in ``fits.py`` function
        signatures).
    rename_params : dict or None
        Optional ``{canonical_name: display_name}`` override map. Wired by
        Phase 13.50 step 3 from ``fit_textbox_kwargs={'rename_params': ...}``.

    Returns
    -------
    str
        The display name to render in the textbox or table.
    """
    if rename_params and canonical in rename_params:
        return rename_params[canonical]
    return _DISPLAY_NAMES.get(canonical, canonical)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

def render_fit_overlays(ax,
                        curves_list: List[Dict[str, Any]],
                        fits_per_curve: List[List[Dict[str, Any]]]) -> None:
    """Draw fit-function overlays on ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    curves_list : list of dict
        Per-curve data dicts (built upstream per §4.3). Each entry must have
        ``x_data``, ``y_data``, ``yerr_data``, ``color``, ``label`` keys.
    fits_per_curve : list of list of dict
        ``fits_per_curve[i]`` = list of per-fit result dicts from
        ``dispatch_fit`` for curve ``i``. Shape mirrors §3.5 canonical form.
    """
    linewidth     = _style_get('fit.linewidth', 1.5)
    linestyle_cyc = _style_get('fit.linestyle_cycle', ['-', '--', '-.', ':'])

    for curve_idx, curve_results in enumerate(fits_per_curve):
        if not curve_results:
            continue
        curve = curves_list[curve_idx]
        x_curve = np.asarray(curve['x_data'], dtype=float)
        color = curve.get('color', None)

        # Evaluate overlay on a dense x-grid within the fit range.
        for fit_idx, result in enumerate(curve_results):
            if result.get('fit_status') != 'ok':
                continue
            func = result.get('function')
            if func is None:
                continue
            x_lo, x_hi = result.get('x_range', (None, None))
            if x_lo is None or not np.isfinite(x_lo):
                x_lo = float(np.nanmin(x_curve)) if x_curve.size else 0.0
            if x_hi is None or not np.isfinite(x_hi):
                x_hi = float(np.nanmax(x_curve)) if x_curve.size else 1.0
            x_eval = np.linspace(x_lo, x_hi, 200)
            try:
                y_eval = func(x_eval)
            except Exception:
                continue
            linestyle = linestyle_cyc[fit_idx % len(linestyle_cyc)]
            ax.plot(x_eval, y_eval,
                    linewidth=linewidth,
                    linestyle=linestyle,
                    color=color)


# ---------------------------------------------------------------------------
# Textbox rendering
# ---------------------------------------------------------------------------

# Canonical position → (x, y, ha, va) in axes-fraction coords
_POSITION_COORDS = {
    'upper left':   (0.02, 0.98, 'left',  'top'),
    'upper right':  (0.98, 0.98, 'right', 'top'),
    'lower left':   (0.02, 0.02, 'left',  'bottom'),
    'lower right':  (0.98, 0.02, 'right', 'bottom'),
    'upper center': (0.50, 0.98, 'center', 'top'),
    'lower center': (0.50, 0.02, 'center', 'bottom'),
    'center left':  (0.02, 0.50, 'left',   'center'),
    'center right': (0.98, 0.50, 'right',  'center'),
    'center':       (0.50, 0.50, 'center', 'center'),
}


def render_fit_textbox(ax,
                       curves_list: List[Dict[str, Any]],
                       fits_per_curve: List[List[Dict[str, Any]]],
                       *,
                       facet_mode: bool = False,
                       textbox_kwargs: Optional[Dict[str, Any]] = None) -> None:
    """Render the ROOT-style fit-parameter textbox on ``ax``.

    Per-fit blocks are concatenated. Each block shows:
        <fit_name>
          <param> = <value> ± <error>
          ...
          χ²/ndf = <chi2>/<ndf> = <redchi>
    Blocks separated by a short separator. Blocks with ``show_params=False``
    are skipped (their overlay was still drawn by ``render_fit_overlays``).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    curves_list, fits_per_curve : see ``render_fit_overlays``.
    facet_mode : bool
        If True, use the smaller facet fontsize (``fit.text_fontsize_facet``,
        default 7). Otherwise use ``fit.text_fontsize_default`` (default 9).
    textbox_kwargs : dict, optional
        Per-call overrides for textbox formatting. Phase 13.42.DF FIX1
        (B2/B3/R5). Accepted sub-keys (names + enum values LOCKED at FIX1 close):
        - 'fontsize' (int): overrides facet/default fontsize style key
        - 'format' (str): 'multiline' | 'compact' | 'auto' (default 'auto').
            'multiline' = one block per fit (header + per-param lines + chi²/ndf)
            'compact'   = one LINE per fit (slot params + chi²/ndf inline)
            'auto'      = compact if facet_mode AND more than 1 fit, else multiline
        - 'show_fields' (list[str]): subset of {'amplitude','center','sigma',
            'slope','intercept','chi2','ndf','redchi','fit_name','x_range'}.
            None or absent → all available fields rendered.
    """
    # ---- Resolve overrides from textbox_kwargs vs style defaults -------
    textbox_kwargs = textbox_kwargs or {}
    _allowed_sub_keys = {'fontsize', 'format', 'show_fields'}
    for _k in textbox_kwargs:
        if _k not in _allowed_sub_keys:
            raise ValueError(
                f"[fit_textbox_kwargs] unknown sub-key {_k!r}. "
                f"Fix: choose from {sorted(_allowed_sub_keys)}."
            )

    position = _style_get('fit.position', 'upper left')
    text_format = _style_get('fit.text_format', '.4g')
    padding = _style_get('fit.text_padding', 0.4)

    # B1/R5 fontsize resolution: per-call override > facet/default style.
    _override_fontsize = textbox_kwargs.get('fontsize')
    if _override_fontsize is not None:
        if not isinstance(_override_fontsize, int) or _override_fontsize <= 0:
            raise ValueError(
                f"[fit_textbox_kwargs] fontsize must be a positive int, "
                f"got {_override_fontsize!r}."
            )
        fontsize = _override_fontsize
    elif facet_mode:
        fontsize = _style_get('fit.text_fontsize_facet', 7)
    else:
        fontsize = _style_get('fit.text_fontsize_default', 9)

    # B2 format resolution: 'auto' → compact when in facet mode with multiple
    # fits (the density case that motivated B2). 'multiline' (v1.0 default)
    # remains the explicit verbose form. 'compact' = one line per fit.
    _format_mode = textbox_kwargs.get('format', 'auto')
    if _format_mode not in ('multiline', 'compact', 'auto'):
        raise ValueError(
            f"[fit_textbox_kwargs] format must be 'multiline'|'compact'|'auto', "
            f"got {_format_mode!r}."
        )
    if _format_mode == 'auto':
        # Count total fit blocks across curves to decide auto threshold
        _n_blocks = sum(len(cr) for cr in fits_per_curve if cr)
        _format_mode = 'compact' if (facet_mode and _n_blocks > 1) else 'multiline'

    # show_fields filter (None → all)
    _show_fields = textbox_kwargs.get('show_fields')
    if _show_fields is not None:
        _valid_fields = {'amplitude', 'center', 'sigma',
                         'slope', 'intercept',
                         'chi2', 'ndf', 'redchi',
                         'fit_name', 'x_range'}
        for _f in _show_fields:
            if _f not in _valid_fields:
                raise ValueError(
                    f"[fit_textbox_kwargs] unknown show_fields entry {_f!r}. "
                    f"Fix: choose from {sorted(_valid_fields)}."
                )
        _show_set = set(_show_fields)
    else:
        _show_set = None  # show all

    def _field_allowed(name: str) -> bool:
        """Check if a field name should be rendered. None set means show all."""
        if _show_set is None:
            return True
        return name in _show_set

    blocks: List[str] = []
    for curve_idx, curve_results in enumerate(fits_per_curve):
        if not curve_results:
            continue
        # Optional group/curve label prefix when there's more than one curve
        curve_label = curves_list[curve_idx].get('label') if curve_idx < len(curves_list) else None

        for result in curve_results:
            # 'show_params' default True. False suppresses the textbox block
            # for this specific fit (overlay still drawn).
            spec = result.get('fit_spec', {})
            if spec.get('show_params', True) is False:
                continue

            label = spec.get('label') or result.get('fit_name', '?')
            if curve_label and len(fits_per_curve) > 1:
                header = f"{label} [{curve_label}]"
            else:
                header = str(label)

            # Failed-fit short-circuit (same for both formats)
            if result.get('fit_status') == 'failed':
                err = result.get('fit_error') or 'no convergence'
                if _format_mode == 'compact':
                    blocks.append(f"{header}: FIT FAILED")
                else:
                    blocks.append(f"{header}\n  FIT FAILED: {err}")
                continue

            params = result.get('params', [])
            perr = result.get('param_errors', [])
            names = result.get('param_names') or [
                f'p{i}' for i in range(len(params))
            ]
            chi2 = result.get('chi2', float('nan'))
            ndf = result.get('ndf', 0)
            redchi = result.get('redchi', float('nan'))

            if _format_mode == 'compact':
                # B2: one line per fit. Pack params + chi²/ndf inline.
                pieces = []
                for i, val in enumerate(params):
                    canonical = names[i] if i < len(names) else f'p{i}'
                    if not _field_allowed(canonical):
                        continue
                    # Phase 13.50: canonical name preserved for _field_allowed
                    # check; display name (short/Greek) used in user-facing
                    # output only. rename_params wired in step 3.
                    name_s = _resolve_display_name(canonical)
                    error = perr[i] if i < len(perr) else float('nan')
                    val_s = format(val, text_format)
                    if np.isfinite(error):
                        err_s = format(error, text_format)
                        pieces.append(f"{name_s}={val_s}±{err_s}")
                    else:
                        pieces.append(f"{name_s}={val_s}")
                if np.isfinite(chi2) and ndf > 0 and (
                    _field_allowed('chi2') or _field_allowed('ndf') or _field_allowed('redchi')
                ):
                    pieces.append(
                        f"χ²/ndf={format(chi2, text_format)}/{ndf}"
                        f"={format(redchi, text_format)}"
                    )
                blocks.append(f"{header}: " + "  ".join(pieces) if pieces else header)
            else:
                # multiline (v1.0 default)
                block_lines = [header]
                for i, val in enumerate(params):
                    canonical = names[i] if i < len(names) else f'p{i}'
                    if not _field_allowed(canonical):
                        continue
                    # Phase 13.50: canonical name preserved for _field_allowed;
                    # display name (short/Greek) used in user-facing output.
                    name_s = _resolve_display_name(canonical)
                    error = perr[i] if i < len(perr) else float('nan')
                    val_s = format(val, text_format)
                    err_s = format(error, text_format) if np.isfinite(error) else '—'
                    block_lines.append(f"  {name_s} = {val_s} ± {err_s}")

                if np.isfinite(chi2) and ndf > 0 and (
                    _field_allowed('chi2') or _field_allowed('ndf') or _field_allowed('redchi')
                ):
                    block_lines.append(
                        f"  χ²/ndf = {format(chi2, text_format)}/{ndf} "
                        f"= {format(redchi, text_format)}"
                    )
                blocks.append('\n'.join(block_lines))

    if not blocks:
        return

    # Separator: compact mode uses just \n, multiline uses the v1.0 hr
    if _format_mode == 'compact':
        full_text = '\n'.join(blocks)
    else:
        full_text = ('\n' + ('─' * 18) + '\n').join(blocks)

    coords = _POSITION_COORDS.get(position, _POSITION_COORDS['upper left'])
    x, y, ha, va = coords
    ax.text(
        x, y, full_text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment=va,
        horizontalalignment=ha,
        family='monospace',
        bbox=dict(
            boxstyle=f'round,pad={padding}',
            facecolor='white',
            edgecolor='gray',
            alpha=0.85,
        ),
    )
