"""PHASE_13_107_DF: histogram X-quantile markers (one owner).

Plain-language purpose
----------------------
``hist(..., quantiles=[...])`` draws one vertical line per requested level
for every drawn histogram curve and returns the X positions.  This module
is the single owner of:

* validating the requested levels and the weighted method (before any
  Figure is created);
* computing the quantiles of one curve from **all its selected, finite X
  values** (independent of the plotted range or the bin edges);
* drawing the vertical lines (only where they fall inside the shown X range)
  and building the per-curve result record ``stats['hist_quantiles']``.

Estimators (Architect decisions 2026-11-10, PHASE_13_107 S0 report):

* no physical weights: NumPy ``method='linear'``;
* weights, ``weighted_quantile_method='inverted_cdf'`` (default, W1): the
  first x whose cumulative weight reaches q*W (NumPy's weighted method);
* weights, ``weighted_quantile_method='linear'`` (W2, Julia StatsBase
  definition): h = q*(W - w1) + w1, linear interpolation of x over the
  cumulative weights; equals NumPy ``linear`` for equal weights.

For both weighted methods: negative weights are refused (before any Figure,
by ``DFDraw.hist``); rows whose x or weight is not finite are dropped
exactly as the histogram drops them; zero-weight rows carry no mass; a curve
without positive mass has undefined quantiles (NaN, no line); equal x values are ordered by weight
so the result does not depend on row order.  Display normalizations
(``norm``, ``hist_norm``) never make a quantile weighted.

Optional text at the top of each drawn line (``quantile_text=``): 'level'
('10%', 'median'), 'value' (the X position; a date on a time axis) or
'both' ('10%: -1.28'); the text has the curve colour and carries
``_dfdraw_quantile_text`` (the numeric level).

The line of each level carries ``_dfdraw_quantile`` (the numeric level),
the identity used by the shared Figure legend; names come from
``plots/_numeric_labels.py`` (distinct names for distinct levels, 'median'
only for exactly 0.5).
"""
import numbers

import numpy as np

from ._numeric_labels import is_median_level, quantile_percent_labels

WEIGHTED_METHODS = ('inverted_cdf', 'linear')
TEXT_MODES = ('level', 'value', 'both')
DEFAULT_WEIGHTED_METHOD = 'inverted_cdf'
POPULATION = 'all_finite_selected_x'

# One linestyle per non-median level; the exact median is solid.
_LEVEL_LINESTYLES = ('--', '-.', ':', (0, (5, 2, 1, 2, 1, 2)), (0, (1, 1)),
                     (0, (8, 3)), (0, (3, 3, 1, 3, 1, 3, 1, 3)), (0, (2, 4)))


def validate_levels(quantiles):
    """Distinct requested levels in input order, or ValueError.

    A non-empty list/tuple/array of real numbers in [0, 1]; booleans, NaN,
    infinity and strings are refused.  A level requested twice is kept once.
    """
    if quantiles is None:
        return None
    if isinstance(quantiles, (str, bytes)) or not hasattr(quantiles, '__len__'):
        raise ValueError(
            'hist quantiles= must be a list of probabilities in [0, 1], '
            f'e.g. [0.05, 0.5, 0.95]; got {quantiles!r}')
    levels = []
    for q in list(quantiles):
        if (isinstance(q, (bool, np.bool_))
                or not isinstance(q, (numbers.Real, np.floating, np.integer))):
            raise ValueError(
                f'hist quantiles= entries must be real numbers; got {q!r}')
        q = float(q)
        if not np.isfinite(q) or q < 0.0 or q > 1.0:
            raise ValueError(
                f'hist quantiles= entries must lie in [0, 1]; got {q!r}')
        if q not in levels:
            levels.append(q)
    if not levels:
        raise ValueError('hist quantiles= must not be empty')
    return levels


def validate_method(method, has_weights):
    """Check ``weighted_quantile_method`` before any Figure (decision D-3)."""
    if method is None:
        return None
    if method not in WEIGHTED_METHODS:
        raise ValueError(
            "weighted_quantile_method must be 'inverted_cdf' or 'linear'; "
            f'got {method!r}')
    if not has_weights:
        raise ValueError(
            'weighted_quantile_method= needs physical weights (weights= or '
            'weights_vector=); without weights the quantiles are the '
            "unweighted NumPy 'linear' quantiles")
    return method


def validate_text(text):
    """Check ``quantile_text`` before any Figure: None or one of TEXT_MODES."""
    if text is None:
        return None
    if not isinstance(text, str) or text not in TEXT_MODES:
        raise ValueError(
            "hist quantile_text= must be None, 'level', 'value' or 'both'; "
            f'got {text!r}')
    return text


def format_value(x, time_axis=False):
    """The X position as text: 4 significant digits, or a date and time on
    a time axis (X in Matplotlib date numbers)."""
    if time_axis:
        import matplotlib.dates as mdates
        return mdates.num2date(x).strftime('%Y-%m-%d %H:%M:%S')
    return f'{x:.4g}'


def marker_text(q, xq, mode, level_names, time_axis=False):
    """Text of one quantile line for ``quantile_text=mode``."""
    level = 'median' if is_median_level(q) else level_names[q]
    if mode == 'level':
        return level
    value = format_value(xq, time_axis)
    return value if mode == 'value' else f'{level}: {value}'


def _sorted_positive(x, w):
    keep = w > 0
    x, w = x[keep], w[keep]
    order = np.lexsort((w, x))   # by x; equal x by weight (row-order free)
    return x[order], w[order]


def weighted_inverted_cdf(x, w, levels):
    """W1: first x with cumulative weight >= q*W (q=0: smallest x)."""
    x, w = _sorted_positive(x, w)
    cum = np.cumsum(w)
    total = cum[-1]
    out = []
    for q in levels:
        if q == 0.0:
            out.append(float(x[0]))
            continue
        k = int(np.searchsorted(cum, q * total * (1.0 - 1e-12), side='left'))
        out.append(float(x[min(k, len(x) - 1)]))
    return out


def weighted_linear(x, w, levels):
    """W2: h = q*(W - w1) + w1, linear in (cumulative weight, x)."""
    x, w = _sorted_positive(x, w)
    if len(x) == 1:
        return [float(x[0])] * len(levels)
    cum = np.cumsum(w)
    return [float(np.interp(q * (cum[-1] - w[0]) + w[0], cum, x))
            for q in levels]


def curve_quantiles(x, weights, levels, method):
    """Quantile record of one histogram curve.

    ``x``: the curve's selected values (any order, NaN/inf allowed and
    dropped).  ``weights``: aligned physical weights or None.
    ``method``: None, 'inverted_cdf' or 'linear' (weighted only).
    """
    x = np.asarray(x, dtype=float)
    if weights is None:
        finite = np.isfinite(x)
        xs = x[finite]
        record = {'levels': list(levels), 'method': 'linear',
                  'weighted': False, 'n': int(xs.size),
                  'n_positive_weight': None, 'weight_sum': None,
                  'population': POPULATION}
        if xs.size == 0:
            record['x'] = [float('nan')] * len(levels)
        else:
            record['x'] = [float(v) for v in
                           np.quantile(xs, levels, method='linear')]
        return record
    w = np.asarray(weights, dtype=float)
    if w.shape != x.shape:
        raise ValueError('hist quantiles: weights are not aligned with the '
                         'values of this curve')
    finite_x = np.isfinite(x)
    xs, ws = x[finite_x], w[finite_x]
    if not np.all(np.isfinite(ws)) or np.any(ws < 0):
        raise ValueError(
            'hist quantiles with weights need finite, non-negative weights; '
            'this curve has negative or non-finite weights')
    method = method or DEFAULT_WEIGHTED_METHOD
    positive = int(np.count_nonzero(ws > 0))
    total = float(ws.sum())
    record = {'levels': list(levels), 'method': method, 'weighted': True,
              'n': int(xs.size), 'n_positive_weight': positive,
              'weight_sum': total, 'population': POPULATION}
    if xs.size == 0 or positive == 0:
        # No values, or no positive mass: the quantiles are undefined for
        # this curve (no line is drawn), as for an empty curve.
        record['x'] = [float('nan')] * len(levels)
        return record
    kernel = (weighted_inverted_cdf if method == 'inverted_cdf'
              else weighted_linear)
    record['x'] = kernel(xs, ws, levels)
    return record


def draw_markers(ax, record, color, shown_range, text=None, time_axis=False):
    """Draw the visible vertical lines of one curve; fill ``marker_visible``.

    A level whose x lies outside ``shown_range`` (the histogram's shown X
    interval) keeps its numeric value but gets no line; the axis limits are
    never changed.  The first line of a level on an axis carries the legend
    name; later curves on the same axis reuse that entry.  With ``text``
    ('level', 'value', 'both') each drawn line also gets its text, written
    vertically near the top of the line in the curve colour.  Several
    curves on one axis (groups, vector branches) alternate the side of the
    line (left, right) and then step down, so that the texts of close lines
    of different curves do not cover each other.
    """
    names = quantile_percent_labels(record['levels'])
    lo, hi = (float(shown_range[0]), float(shown_range[1]))
    already = {getattr(line, '_dfdraw_quantile', None) for line in ax.lines}
    visible = []
    styles = iter(_LEVEL_LINESTYLES * 8)
    curve = len({getattr(t, '_dfdraw_quantile_curve') for t in ax.texts
                 if hasattr(t, '_dfdraw_quantile_curve')})
    side = 'right' if curve % 2 == 0 else 'left'
    top = 0.98 - 0.4 * ((curve // 2) % 2)
    for q, xq in zip(record['levels'], record['x']):
        style = '-' if is_median_level(q) else next(styles)
        show = bool(np.isfinite(xq) and lo <= xq <= hi)
        visible.append(show)
        if not show:
            continue
        label = ('_nolegend_' if q in already else
                 ('median' if is_median_level(q) else f'q={names[q]}'))
        line = ax.axvline(xq, color=color, linestyle=style, linewidth=1.3,
                          alpha=0.9, zorder=4, label=label)
        line._dfdraw_quantile = q
        already.add(q)
        if text is not None:
            label_artist = ax.text(
                xq, top, marker_text(q, xq, text, names, time_axis),
                transform=ax.get_xaxis_transform(), rotation=90,
                ha=side, va='top', fontsize=7, color=color, zorder=5,
                clip_on=True,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none',
                          pad=0.5))
            label_artist._dfdraw_quantile_text = q
            label_artist._dfdraw_quantile_curve = curve
    record['marker_visible'] = visible
    return record


def shown_range(bins, used_range):
    """The X interval the histogram shows: explicit edges win over range."""
    if bins is not None and not isinstance(bins, (int, np.integer, str)):
        edges = np.asarray(bins, dtype=float)
        if edges.ndim == 1 and edges.size >= 2:
            return float(edges[0]), float(edges[-1])
    return float(used_range[0]), float(used_range[1])
