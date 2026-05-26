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
# DEFAULT_STYLE in style.py; we read it lazily so this module stays cheap).
def _style_get(key: str, default):
    try:
        from ..style import get_style
        return get_style(key)
    except Exception:
        try:
            from dfdraw.style import get_style
            return get_style(key)
        except Exception:
            return default


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
                       facet_mode: bool = False) -> None:
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
    """
    position = _style_get('fit.position', 'upper left')
    text_format = _style_get('fit.text_format', '.4g')
    padding = _style_get('fit.text_padding', 0.4)
    if facet_mode:
        fontsize = _style_get('fit.text_fontsize_facet', 7)
    else:
        fontsize = _style_get('fit.text_fontsize_default', 9)

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

            block_lines: List[str] = []
            label = spec.get('label') or result.get('fit_name', '?')
            if curve_label and len(fits_per_curve) > 1:
                block_lines.append(f"{label} [{curve_label}]")
            else:
                block_lines.append(str(label))

            if result.get('fit_status') == 'failed':
                err = result.get('fit_error') or 'no convergence'
                block_lines.append(f"  FIT FAILED: {err}")
                blocks.append('\n'.join(block_lines))
                continue

            params = result.get('params', [])
            perr = result.get('param_errors', [])
            names = result.get('param_names') or [
                f'p{i}' for i in range(len(params))
            ]
            for i, val in enumerate(params):
                error = perr[i] if i < len(perr) else float('nan')
                # Use the configured format string for both value and error
                val_s = format(val, text_format)
                err_s = format(error, text_format) if np.isfinite(error) \
                    else '—'
                # Right-pad the name for column alignment
                name_s = names[i] if i < len(names) else f'p{i}'
                block_lines.append(f"  {name_s} = {val_s} ± {err_s}")

            chi2 = result.get('chi2', float('nan'))
            ndf = result.get('ndf', 0)
            redchi = result.get('redchi', float('nan'))
            if np.isfinite(chi2) and ndf > 0:
                block_lines.append(
                    f"  χ²/ndf = {format(chi2, text_format)}/{ndf} "
                    f"= {format(redchi, text_format)}"
                )
            blocks.append('\n'.join(block_lines))

    if not blocks:
        return

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
