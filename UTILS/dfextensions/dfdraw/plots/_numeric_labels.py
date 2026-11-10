"""PHASE_13_108_DF v0.2: collision-free text for numeric identities.

Plain-language purpose
----------------------
dfdraw writes numbers into legends, titles and group/facet names: quantile
levels (``q=10.1%``), the quantile list of the automatic title
(``q [0.1, 0.5]``) and numeric bin intervals (``0.91-1.03``).  Short text is
easier to read, but two *different* numbers must never be written the same
way: the legend would merge two different curves, and a group or facet name
that is used as a grouping key would merge two different data intervals.

Every formatter here therefore takes the **whole set** of values that appear
together, keeps today's short text where it is already unambiguous, and
raises the precision only where two different values would otherwise look
identical.  The last resort is the shortest exact decimal text of the float
(``repr``), which is unique for every distinct float.

This module imports nothing from dfdraw (no import cycles).
"""
from decimal import Decimal, ROUND_HALF_EVEN

__all__ = [
    'is_median_level', 'quantile_percent_text', 'quantile_percent_labels',
    'quantile_fraction_labels', 'interval_labels',
]

# Today's short forms: six significant digits for quantile levels.
_DEFAULT_DIGITS = 6
_MAX_DIGITS = 17


def is_median_level(q):
    """True only for the exact probability 0.5.

    A nearby but different level (0.5000001) is a different requested
    quantile and keeps its own label and legend entry.
    """
    return float(q) == 0.5


def _exact_decimal(value):
    """The shortest decimal that round-trips to this float, as Decimal."""
    return Decimal(repr(float(value)))


def _significant_text(value, digits):
    """Fixed-point text of ``value`` rounded to ``digits`` significant digits,
    without trailing zeros; ``digits=None`` gives the exact decimal."""
    d = _exact_decimal(value) if not isinstance(value, Decimal) else value
    if d == 0:
        return '0'
    if digits is not None:
        exponent = d.adjusted() - digits + 1
        d = d.quantize(Decimal(1).scaleb(exponent), rounding=ROUND_HALF_EVEN)
    text = format(d, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text or '0'


def _distinct_texts(values, render, reserved=()):
    """Map each distinct value to ``render(value, digits)``; start at the
    default precision and raise it only for values whose text collides.

    ``reserved`` values take part in the collision check without being
    returned: a level 0.5000000001 must not print as the '50%' of the
    exact median.
    """
    requested = sorted({float(v) for v in values})
    values = sorted(set(requested) | {float(v) for v in reserved})
    texts = {v: render(v, _DEFAULT_DIGITS) for v in values}
    digits = _DEFAULT_DIGITS
    while True:
        by_text = {}
        for v in values:
            by_text.setdefault(texts[v], []).append(v)
        colliding = [v for group in by_text.values() if len(group) > 1
                     for v in group]
        if not colliding:
            break
        digits += 1
        for v in colliding:
            texts[v] = render(v, digits if digits <= _MAX_DIGITS else None)
        if digits > _MAX_DIGITS:
            break
    return {v: texts[v] for v in requested}


def quantile_percent_text(q, digits=_DEFAULT_DIGITS):
    """One quantile level as a percent: 0.05 -> '5%', 0.101 -> '10.1%'."""
    return _significant_text(_exact_decimal(q) * 100, digits) + '%'


def quantile_percent_labels(levels):
    """``{float(q): '10.1%'}`` for every requested level, distinct texts for
    distinct levels (0.100000001 and 0.100000002 are both kept apart).  A
    level that is not exactly 0.5 never prints as '50%'."""
    return _distinct_texts(
        levels, lambda q, d: _significant_text(_exact_decimal(q) * 100, d) + '%',
        reserved=(0.5,))


def quantile_fraction_labels(levels):
    """``{float(q): '0.101'}``: the same rule in fraction form (auto title)."""
    return _distinct_texts(
        levels, lambda q, d: _significant_text(_exact_decimal(q), d),
        reserved=(0.5,))


def _interval_decimals(interval):
    """PHASE_13_12 AD-3 rule: decimals chosen from the interval width."""
    width = interval.right - interval.left
    if width >= 10:
        return 0
    if width >= 1:
        return 1
    if width >= 0.1:
        return 2
    return 3


def _interval_text(interval, decimals):
    return f"{interval.left:.{decimals}f}-{interval.right:.{decimals}f}"


def _interval_texts_ambiguous(intervals, texts):
    """Two different intervals with one text, or an interval whose two
    different edges print identically ('0.000-0.000')."""
    if len(set(texts)) != len(texts):
        return True
    for interval, text in zip(intervals, texts):
        if interval.left != interval.right:
            left, right = _edge_texts(text)
            if left == right:
                return True
    return False


def _edge_texts(text):
    # 'a-b' where a and b may be negative ('-0.39--0.00'): split at the
    # separator dash, which follows a digit.
    for i in range(1, len(text)):
        if text[i] == '-' and text[i - 1].isdigit():
            return text[:i], text[i + 1:]
    return text, ''


def interval_labels(intervals):
    """``{interval: 'low-high'}`` for the bins of one variable.

    Today's per-interval text (decimals from the interval width) is kept
    whenever it is unambiguous, so existing group and facet names do not
    change.  If two bins would get the same name, or a bin's two edges
    would print identically, all bins of the set get more decimals until
    every name is distinct.
    """
    intervals = list(intervals)
    texts = [_interval_text(iv, _interval_decimals(iv)) for iv in intervals]
    if not _interval_texts_ambiguous(intervals, texts):
        return dict(zip(intervals, texts))
    decimals = max((_interval_decimals(iv) for iv in intervals), default=3)
    while decimals < 15:
        decimals += 1
        texts = [_interval_text(iv, decimals) for iv in intervals]
        if not _interval_texts_ambiguous(intervals, texts):
            return dict(zip(intervals, texts))
    return {iv: f"{_significant_text(iv.left, None)}-"
                f"{_significant_text(iv.right, None)}" for iv in intervals}


def map_interval_labels(classified):
    """Name the rows of a ``pd.cut``/``pd.qcut`` result with
    :func:`interval_labels` of its categories (same shape as the former
    ``classified.map(_format_interval_label)``)."""
    return classified.map(interval_labels(classified.cat.categories))
