"""Regression tests — bug catalog instance #9 warning (Phase 13.23c.GB step 3).

Architect ruling 2026-07-04 (verbatim): "OK. Let's make it a warning."
The recompute path (default algorithm) of make_sliding_window_fit does not
consume `boundary`; computation proceeds as boundary='full'. This suite
locks: the loud UserWarning (W1), its absence where behaviour is correct
(W2, W3), truth-in-metadata (W4), and that the warning changes no numbers
(W5). All tests call `make_sliding_window_fit` directly (entry point
exercised, failure-mode #12 discipline).
"""
import sys
import os
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from groupby_regression_sliding_window import make_sliding_window_fit  # noqa: E402

RNG = np.random.default_rng(99)


def _fixture(n_x=4, n_y=4, rows_per_bin=25):
    rows = []
    for bx in range(n_x):
        for by in range(n_y):
            x = RNG.normal(size=rows_per_bin)
            y = 0.7 * x + 0.1 * bx - 0.2 * by + RNG.normal(
                scale=0.05, size=rows_per_bin)
            rows.append(pd.DataFrame(
                {"bin_x": bx, "bin_y": by, "x": x, "y": y}))
    return pd.concat(rows, ignore_index=True)


_KW = dict(gb_columns=["bin_x", "bin_y"], fit_columns=["y"],
           linear_columns=["x"], window_spec={"bin_x": 1, "bin_y": 1},
           min_stat=5, suffix="_sw")


def test_w1_warning_fires_symmetric_default_algorithm():
    df = _fixture()
    with pytest.warns(UserWarning, match="IGNORED"):
        make_sliding_window_fit(df=df, boundary="symmetric", **_KW)


def test_w1b_warning_fires_for_dict_with_any_nonfull_dim():
    df = _fixture()
    with pytest.warns(UserWarning, match="IGNORED"):
        make_sliding_window_fit(
            df=df, boundary={"bin_x": "symmetric", "bin_y": "full"}, **_KW)


def test_w2_no_warning_full_default_algorithm():
    df = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        make_sliding_window_fit(df=df, boundary="full", **_KW)


def test_w3_no_warning_symmetric_incremental():
    df = _fixture()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_sliding_window_fit(
            df=df, boundary="symmetric", algorithm="incremental", **_KW)


def test_w4_metadata_reports_actual_boundary_full_on_recompute():
    df = _fixture()
    with pytest.warns(UserWarning, match="IGNORED"):
        out, meta = make_sliding_window_fit(
            df=df, boundary="symmetric", return_metadata=True,
            backend="numpy", **_KW)  # numpy backend -> shared metadata block
    bm = meta["boundary_mode"]
    assert all(v == "full" for v in bm.values()), (
        f"metadata must record the ACTUAL boundary ('full'), got {bm}")
    # attrs on the frame must agree with the returned metadata
    assert out.attrs["boundary_mode"] == bm


def test_w4b_metadata_reports_requested_boundary_on_incremental():
    """Control: on the honouring path, metadata keeps the requested mode."""
    df = _fixture()
    out, meta = make_sliding_window_fit(
        df=df, boundary="symmetric", algorithm="incremental",
        backend="numpy", return_metadata=True, **_KW)
    assert all(v == "symmetric" for v in meta["boundary_mode"].values())


def test_w5_warning_changes_no_numbers():
    """The warned symmetric+recompute output is bit-identical to the
    full+recompute output — i.e. current (ignored) behaviour is unchanged
    by the warning. Together with the pristine-module comparison in the
    CRR run log this is the captured-baseline requirement of the work
    order."""
    df = _fixture()
    with pytest.warns(UserWarning, match="IGNORED"):
        warned = make_sliding_window_fit(
            df=df, boundary="symmetric", backend="numpy", **_KW)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        silent = make_sliding_window_fit(
            df=df, boundary="full", backend="numpy", **_KW)
    pd.testing.assert_frame_equal(warned, silent,
                                  check_exact=True, check_dtype=True)
