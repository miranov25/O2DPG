"""Shared time-coordinate conversion owner for dfdraw.

BUGFIX time_format/date2num performance (2026-09-10).

Single semantic owner for converting a user x/y vector into matplotlib date
coordinates when ``time_format=`` is set.  Before this module the same
conversion was copy-pasted at eight call sites across ``plots/histogram.py``,
``plots/profile.py`` and ``plots/scatter.py``; each of them ran

    mdates.date2num(pd.to_datetime(arr, unit='s').to_pydatetime())

which materialises one Python ``datetime`` object per input row.  On a
9.8-million-row production frame that cost ~22 s per converted vector and
dominated the ADF FULL+LAZY gallery profile (bug report
BUG_dfdraw_20260910_time_format_full_array_date2num_performance).

The public ``time_format=`` contract is unchanged.  Inputs are interpreted
exactly as before:

    datetime64 dtype  -> converted directly by matplotlib
    anything else     -> interpreted as Unix epoch SECONDS

Conversion strategy (numeric input):

    1. Fast path -- pure float arithmetic.  Matplotlib date numbers are days
       since ``matplotlib.dates.get_epoch()``, so epoch seconds convert with
       one divide and one add.  The epoch offset is queried from matplotlib at
       call time, so a non-default ``rcParams['date.epoch']`` is honoured.
    2. Fallback -- the original pandas route, used whenever any finite input
       lies outside the datetime64[ns] representable range.  Out-of-range input
       therefore still raises the same pandas exception it raised before, so
       error behaviour does not silently change.

Non-finite values (NaN/inf) propagate to NaN date coordinates, matching what
matplotlib produces for NaT.  In practice dfdraw sanitises x/y before calling
here, so this is defensive only.
"""

import numpy as np
import pandas as pd


# datetime64[ns] representable range expressed in Unix epoch seconds.
# pd.Timestamp.min / .max are 1677-09-21 and 2262-04-11.
_NS_MIN_SECONDS = -9.2233720368e9
_NS_MAX_SECONDS = 9.2233720368e9

_SECONDS_PER_DAY = 86400.0


def _epoch_offset_days():
    """Matplotlib date number of the Unix epoch, queried at call time.

    Returned as a float so the caller can convert epoch seconds with plain
    arithmetic.  Queried (not hard-coded) so that a non-default
    ``rcParams['date.epoch']`` is respected.
    """
    import matplotlib.dates as mdates
    return float(mdates.date2num(np.datetime64("1970-01-01T00:00:00.000000", "us")))


def _in_ns_range(values):
    """True when every finite entry is representable as datetime64[ns]."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return True
    return bool(finite.min() >= _NS_MIN_SECONDS and finite.max() <= _NS_MAX_SECONDS)


def to_date_coords(values):
    """Convert a value vector to matplotlib date coordinates.

    Parameters
    ----------
    values : array-like
        Either a datetime64 array, or numeric Unix epoch seconds.

    Returns
    -------
    numpy.ndarray
        Float array of matplotlib date numbers (ordinal days).

    Notes
    -----
    This is the single conversion owner for ``time_format=``.  Plot families
    must call it rather than re-implementing the conversion.
    """
    import matplotlib.dates as mdates

    arr = np.asarray(values)

    # datetime64 input: matplotlib already handles this vectorised, and the
    # epoch-seconds reinterpretation must NOT be applied (the int64 nanosecond
    # representation would be read as seconds -> year out of range).
    if np.issubdtype(arr.dtype, np.datetime64):
        return mdates.date2num(arr)

    numeric = arr.astype(float, copy=False)

    # Fallback to the ORIGINAL expression for out-of-range input, so whatever
    # that path did before (including the exception it raised) is unchanged.
    # Out-of-range input is by definition pathological, so its cost is
    # irrelevant; only the normal path needs to be fast.
    if not _in_ns_range(numeric):
        return mdates.date2num(pd.to_datetime(arr, unit="s").to_pydatetime())

    return _epoch_offset_days() + numeric / _SECONDS_PER_DAY
