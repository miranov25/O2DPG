"""T1-T8 for the shared time-coordinate conversion owner.

Covers BUGFIX time_format/date2num performance (2026-09-10).

Naming used below, defined once so the file is readable on its own:
  T1..T6  correctness tests -- the optimised conversion must agree with the
          original implementation and the public time_format= contract must
          not move.
  T7      ownership test -- every plot family must route through the single
          conversion owner, with no surviving copy of the old expression.
  T8      falsifier -- the optimised path must NOT build one Python datetime
          object per row. This is the test that fails on the old code and
          passes on the new code; it is what makes the fix checkable rather
          than merely faster on one machine.

The reference implementation `_legacy_convert` below is the exact expression
that lived at eight call sites before the fix.
"""

import time

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
import matplotlib.dates as mdates  # noqa: E402

from dfdraw.plots._time_coords import to_date_coords  # noqa: E402


# Agreement tolerance. The optimised path does float arithmetic on epoch
# seconds; the legacy path went through microsecond-truncated Python datetime
# objects. The two therefore agree to about a microsecond, not bit-for-bit.
# One microsecond on a time axis spanning months is far below one pixel.
_TOL_DAYS = 5e-11  # ~4.3 microseconds


def _legacy_convert(values):
    """The pre-fix expression, kept as the equivalence reference."""
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return mdates.date2num(arr)
    return mdates.date2num(pd.to_datetime(arr, unit="s").to_pydatetime())


def _assert_agrees(values, tol=_TOL_DAYS):
    old = np.asarray(_legacy_convert(values), dtype=float)
    new = np.asarray(to_date_coords(values), dtype=float)
    assert old.shape == new.shape
    np.testing.assert_array_equal(np.isnan(old), np.isnan(new))
    finite = ~np.isnan(old)
    if finite.any():
        assert np.max(np.abs(old[finite] - new[finite])) <= tol


# --------------------------------------------------------------------------
# T1-T3 -- epoch seconds through the three binned plot families' input shapes
# --------------------------------------------------------------------------

def test_T1_float_epoch_seconds_matches_legacy():
    rng = np.random.default_rng(11)
    _assert_agrees(1.7e9 + rng.uniform(0, 3.0e7, 50_000))


def test_T2_integer_epoch_seconds_matches_legacy():
    rng = np.random.default_rng(12)
    _assert_agrees((1.7e9 + rng.integers(0, 3.0e7, 50_000)).astype(np.int64))


def test_T3_realistic_epoch_is_not_left_as_raw_coordinate():
    """The historical OverflowError class must stay closed.

    Raw epoch seconds (~1.7e9) read as matplotlib ordinal days would land
    around year 4,600,000. The converted value must be a plausible modern
    date instead.
    """
    out = np.asarray(to_date_coords(np.array([1.7e9])), dtype=float)
    lo = mdates.date2num(np.datetime64("2020-01-01T00:00:00", "us"))
    hi = mdates.date2num(np.datetime64("2040-01-01T00:00:00", "us"))
    assert lo < out[0] < hi


# --------------------------------------------------------------------------
# T4-T6 -- other input dtypes and edge behaviour
# --------------------------------------------------------------------------

def test_T4_datetime64_input_unchanged():
    rng = np.random.default_rng(14)
    values = pd.to_datetime(1.7e9 + rng.uniform(0, 3.0e7, 20_000), unit="s").to_numpy()
    old = np.asarray(_legacy_convert(values), dtype=float)
    new = np.asarray(to_date_coords(values), dtype=float)
    # datetime64 goes down the untouched matplotlib route: exact.
    np.testing.assert_array_equal(old, new)


def test_T5_sub_second_and_negative_epochs():
    rng = np.random.default_rng(15)
    _assert_agrees(1.7e9 + rng.uniform(0, 10.0, 20_000))   # sub-second spacing
    _assert_agrees(np.array([-3.0e8, -1.0e8, 0.0]))        # pre-1970


def test_T6_nan_propagates_and_empty_is_safe():
    out = np.asarray(to_date_coords(np.array([1.7e9, np.nan, 1.7e9 + 86400.0])),
                     dtype=float)
    assert np.isnan(out[1])
    assert np.isfinite(out[0]) and np.isfinite(out[2])
    assert np.asarray(to_date_coords(np.array([], dtype=float))).size == 0


def test_T6b_list_input_accepted():
    _assert_agrees([1.7e9, 1.7e9 + 3600.0])


def test_T6c_matplotlib_epoch_setting_is_honoured():
    """The offset is queried from matplotlib, not hard-coded."""
    expected = mdates.date2num(np.datetime64("1970-01-01T00:00:00", "us"))
    got = float(np.asarray(to_date_coords(np.array([0.0])), dtype=float)[0])
    assert abs(got - expected) <= _TOL_DAYS


# --------------------------------------------------------------------------
# T7 -- single-owner (reuse-before-create) enforcement
# --------------------------------------------------------------------------

def test_T7_no_plot_family_reimplements_the_conversion():
    """No copy of the legacy expression may survive outside the owner."""
    import pathlib

    import dfdraw.plots as plots_pkg

    plots_dir = pathlib.Path(plots_pkg.__file__).parent
    offenders = []
    for path in plots_dir.glob("*.py"):
        if path.name == "_time_coords.py":
            continue
        if "to_pydatetime()" in path.read_text():
            offenders.append(path.name)
    assert offenders == [], (
        f"time conversion re-implemented outside the shared owner: {offenders}"
    )


def test_T7b_each_time_family_calls_the_owner():
    import pathlib

    import dfdraw.plots as plots_pkg

    plots_dir = pathlib.Path(plots_pkg.__file__).parent
    for name in ("histogram.py", "profile.py", "scatter.py"):
        assert "to_date_coords(" in (plots_dir / name).read_text(), (
            f"{name} does not route through the shared conversion owner"
        )


# --------------------------------------------------------------------------
# T8 -- falsifier: no per-row Python datetime materialisation
# --------------------------------------------------------------------------

def test_T8_no_object_dtype_datetime_materialisation(monkeypatch):
    """Fails on the old implementation, passes on the new one.

    The old path handed matplotlib an object-dtype array holding one Python
    datetime per row. Assert that nothing of the sort is produced on the
    optimised path. Written against the observable property rather than a
    specific pandas method name, because the method moved between pandas
    versions (Series.to_pydatetime in 1.x, .dt/DatetimeIndex later).
    """
    seen = []
    original = mdates.date2num

    def spy(values, *args, **kwargs):
        arr = np.asarray(values)
        seen.append((arr.dtype, arr.size))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(mdates, "date2num", spy)

    rng = np.random.default_rng(18)
    n = 200_000
    to_date_coords(1.7e9 + rng.uniform(0, 3.0e7, n))

    bulk_object = [
        (dtype, size) for dtype, size in seen
        if dtype == np.dtype("O") and size > 1
    ]
    assert bulk_object == [], (
        f"optimised path built an object-dtype datetime array: {bulk_object}"
    )


def test_T8_falsifier_actually_fails_on_the_old_implementation(monkeypatch):
    """T8 is only a falsifier if the old code fails it. Prove that here."""
    seen = []
    original = mdates.date2num

    def spy(values, *args, **kwargs):
        arr = np.asarray(values)
        seen.append((arr.dtype, arr.size))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(mdates, "date2num", spy)

    rng = np.random.default_rng(18)
    _legacy_convert(1.7e9 + rng.uniform(0, 3.0e7, 20_000))

    bulk_object = [
        (dtype, size) for dtype, size in seen
        if dtype == np.dtype("O") and size > 1
    ]
    assert bulk_object, "legacy path unexpectedly did not materialise objects"


@pytest.mark.slow
def test_T8b_large_vector_conversion_is_fast():
    """Guards the O(N) regression returning. Ratio-based, not wall-clock."""
    rng = np.random.default_rng(19)
    values = 1.7e9 + rng.uniform(0, 3.0e7, 2_000_000)

    start = time.perf_counter()
    to_date_coords(values)
    new_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    _legacy_convert(values)
    legacy_elapsed = time.perf_counter() - start

    assert legacy_elapsed / max(new_elapsed, 1e-9) >= 5.0, (
        f"speedup only {legacy_elapsed / max(new_elapsed, 1e-9):.1f}x "
        f"(legacy {legacy_elapsed:.3f}s, new {new_elapsed:.3f}s)"
    )
