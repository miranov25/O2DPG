"""PHASE_13_101.ADF F1: regression for the historical injected-residual oracle.

This is an isolated synthetic test; it does not load ROOT data or change
production ADF/dfdraw.  The real gallery runs separately after installation.

The public profile plots ``oracle_residual`` after stepwise dtype coercion, NOT
ideal float64 ``oracle_noise``.  A strong test must (1) distinguish the two,
(2) accept the reconstructed residual and (3) reject a corrupted public mean.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


_HARNESS_PATH = (Path(__file__).resolve().parents[1]
                 / "examples" / "time_series" / "time_series_draw_invariance.py")


def _harness():
    """Import the actual installed ADF harness without requiring a ROOT file."""
    name = "_adf101_f1_actual_harness"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, _HARNESS_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _synthetic_adf(h):
    """Two occupied sector/side cells with deliberately visible float32 loss.

    Build each alias from physical source values in the documented sequence.
    The injected Gaussian remains float64, but the authoritative intermediate
    aliases are float32, as in the observed first-million-entry failure.
    """
    ids = np.arange(100, 116, dtype=np.int64)
    side = np.repeat(np.array([0, 1], dtype=np.int8), 8)
    sector = np.where(side == 0, 10.0, 11.0).astype(np.float32)
    tgl = np.where(side == 0, 0.22, -0.34).astype(np.float32)
    dcar = np.where(side == 0, 8.50, 8.25).astype(np.float32)
    noise = h._it_stable_noise(ids)
    df = pd.DataFrame({
        "oracle_row_id": ids, "oracle_noise": noise,
        "sector": sector, "tgl": tgl, "dcar_tpc_vertex": dcar,
        "side_type": side, "ncl": np.full(len(ids), 90, dtype=np.int32),
    })
    delta = (0.08 * np.sin(2.0 * np.pi * sector / 36.0)
             + 0.03 * tgl + 0.015 * tgl * tgl)
    df["known_delta"] = np.asarray(delta).astype(np.float32)
    df["dcar_distorted_clean"] = (
        dcar + df["known_delta"].to_numpy()).astype(np.float32)
    df["dcar_oracle"] = (
        df["dcar_distorted_clean"].to_numpy() + noise).astype(np.float32)
    df["oracle_residual"] = (
        df["dcar_oracle"].to_numpy() - dcar
        - df["known_delta"].to_numpy()).astype(np.float32)
    return SimpleNamespace(df=df)


def _public_frame_from_materialized(adf, facet, *, mutate=False):
    """Independent bin counting and averaging of the plotted stored column."""
    df = adf.df
    x = df["sector"].to_numpy(dtype=np.float64)
    y = df["oracle_residual"].to_numpy(dtype=np.float64)
    eligible = ((df["ncl"].to_numpy() > 60)
                & (np.abs(df["dcar_tpc_vertex"].to_numpy()) < 10)
                & (df["side_type"].to_numpy() == facet)
                & np.isfinite(x) & np.isfinite(y))
    edges = np.linspace(-0.5, 35.5, 37)
    counts = np.zeros(36, dtype=np.int64)
    means = np.full(36, np.nan, dtype=np.float64)
    for i in range(36):
        inside = eligible & (x >= edges[i]) & (x < edges[i + 1]
                    if i < 35 else x <= edges[i + 1])
        counts[i] = int(np.count_nonzero(inside))
        if counts[i]:
            means[i] = float(np.mean(y[inside], dtype=np.float64))
    if mutate:
        occupied = np.flatnonzero(counts)
        assert occupied.size
        means[occupied[0]] += 1e-3
    return pd.DataFrame({
        "x_center": (edges[:-1] + edges[1:]) / 2.0,
        "count": counts,
        "y_mean": means,
    })


class _Gallery:
    def __init__(self, *, mutate=False):
        self.mutate = mutate

    def _ensure_injected_truth(self, adf):
        # Synthetic fixture already contains the reviewed aliases and row IDs.
        assert "oracle_residual" in adf.df

    def fig51_injected_truth_facet_residual(self, adf):
        stats = {
            "groups": [0, 1],
            "per_group": {
                str(f): {"profile_data": _public_frame_from_materialized(
                    adf, f, mutate=self.mutate and f == 0)}
                for f in (0, 1)
            },
        }
        return (plt.figure(), None, stats)


def test_facet_oracle_uses_dtype_aware_residual_not_ideal_noise(monkeypatch):
    h = _harness()
    monkeypatch.setattr(h, "plt", plt, raising=False)
    adf = _synthetic_adf(h)
    independent = h._it_semantic_model(adf)

    # Stepwise independent computation agrees with stored alias values, but
    # deliberately differs from ideal Gaussian noise at strict tolerance.
    np.testing.assert_array_equal(
        independent["residual"], adf.df["oracle_residual"].to_numpy())
    assert h._it_raw_model(adf)["truth_domain_rows"] == len(adf.df)
    expected, observed = h._it_facet(adf, _Gallery())
    np.testing.assert_array_equal(expected["count"], observed["count"])
    h._it_close("stored-residual facet mean", expected["y_mean"], observed["y_mean"])

    ideal_facet_zero = h._it_profile(
        independent["sector"], independent["noise"],
        independent["side_sel"] & (independent["side"] == 0))
    actual_bin_10 = float(observed["y_mean"][10])
    assert not np.isclose(ideal_facet_zero["y_mean"][10], actual_bin_10,
                          rtol=1e-8, atol=1e-9)
    assert "dtype-aware" in h.injected_truth_facet_case().figure_contract.primary_comparison


def test_facet_oracle_still_detects_a_wrong_public_mean(monkeypatch):
    h = _harness()
    monkeypatch.setattr(h, "plt", plt, raising=False)
    expected, corrupted = h._it_facet(_synthetic_adf(h), _Gallery(mutate=True))
    np.testing.assert_array_equal(expected["count"], corrupted["count"])
    with pytest.raises(h.HarnessError, match="numerical mismatch"):
        h._it_close("corrupted facet mean", expected["y_mean"], corrupted["y_mean"])
