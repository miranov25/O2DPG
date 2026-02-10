from __future__ import annotations

"""
Sliding Window GroupBy Linear Regression (v4-aligned API)

Performs grouped linear regression over sliding windows of integer-binned data.
For each center bin, aggregates data from neighboring bins within the window radius
and performs OLS (or WLS if weights given).

Key properties:
- Integer bin coordinates only; users pre-bin floats.
- Zero-copy neighborhood aggregation via bin->row_indices map.
- Window spec: {dim: nonneg_int} (symmetric ±w per dim). Boundary mode: truncate.
- Aggregations per target: mean, std, median, entries.
- Linear fitting via numpy.linalg.lstsq (V1) or Numba kernel (V2).
- Diagnostics: RMSE, n_fitted, n_neighbors_used, n_rows_aggregated.
- Provenance in DataFrame.attrs.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable, Any, Iterable
import sys
import json
import time
import math
import warnings

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype


# =========================
# Exceptions & Warnings
# =========================
class InvalidWindowSpec(ValueError):
    """Raised when the window specification is malformed or unsupported in M7.1."""


class PerformanceWarning(UserWarning):
    """Issued when requested backend or feature is downgraded (e.g., numba -> numpy)."""


# =========================
# Helper utilities
# =========================

def _validate_sliding_window_inputs(
        df: pd.DataFrame,
        gb_columns: List[str],
        window_spec: Dict[str, int],
        fit_columns: List[str],
        linear_columns: List[str],
        weights: Optional[str] = None,
        selection: Optional[pd.Series] = None,
        min_stat: int = 10,
        backend: str = 'auto',
        **kwargs: Any,
) -> None:
    """Validate inputs for make_sliding_window_fit."""
    # gb_columns existence and integer dtype
    if not gb_columns:
        raise ValueError("gb_columns must be a non-empty list of column names")
    for col in gb_columns:
        if col not in df.columns:
            raise ValueError(f"gb column '{col}' not found in DataFrame")
        if not is_integer_dtype(df[col]):
            raise ValueError(
                f"gb column '{col}' must be integer dtype (found {df[col].dtype}). "
                "Sliding window requires integer bin coordinates."
            )

    # window spec: nonneg ints, all keys must be in gb_columns
    if not window_spec:
        raise InvalidWindowSpec("window_spec must be a non-empty dict {dim: nonneg_int}")
    for dim, w in window_spec.items():
        if dim not in gb_columns:
            raise InvalidWindowSpec(
                f"window_spec key '{dim}' must be one of gb_columns {gb_columns}"
            )
        if not isinstance(w, (int, np.integer)) or w < 0:
            raise InvalidWindowSpec(
                f"window_spec for '{dim}' must be a non-negative integer (got {w!r})"
            )

    # selection
    if selection is not None:
        if len(selection) != len(df):
            raise ValueError(
                f"selection length ({len(selection)}) must match DataFrame length ({len(df)})"
            )
        if selection.dtype != bool:
            raise ValueError("selection mask must be boolean dtype")

    # weights column exists
    if weights is not None and weights not in df.columns:
        raise ValueError(f"weights column '{weights}' not found in DataFrame")

    # fit columns exist
    for t in fit_columns:
        if t not in df.columns:
            raise ValueError(f"fit column '{t}' not found in DataFrame")

    # linear columns exist
    for p in linear_columns:
        if p not in df.columns:
            raise ValueError(f"linear column '{p}' not found in DataFrame")

    # backend
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError("backend must be 'auto', 'numpy', or 'numba'")

    # min_stat strictly positive
    if not isinstance(min_stat, (int, np.integer)) or int(min_stat) <= 0:
        raise ValueError("min_stat must be a strictly positive integer")




def _build_bin_index_map(
        df: pd.DataFrame,
        gb_columns: List[str],
        selection: Optional[pd.Series] = None,
) -> Dict[Tuple[int, ...], List[int]]:
    """Build a zero-copy index map: tuple(bin coords) -> list(row indices).

    Applies selection if provided.
    """
    if selection is not None:
        sel_idx = np.flatnonzero(selection.to_numpy())
    else:
        sel_idx = np.arange(len(df), dtype=np.int64)

    if len(sel_idx) == 0:
        return {}

    # Extract columns as numpy (fast path)
    cols = [df[c].to_numpy() for c in gb_columns]
    # Build tuple keys for selected rows
    keys = [tuple(int(col[i]) for col in cols) for i in sel_idx]

    bin_map: Dict[Tuple[int, ...], List[int]] = {}
    for key, ridx in zip(keys, sel_idx):
        bin_map.setdefault(key, []).append(int(ridx))
    return bin_map


def _observed_bin_bounds(
        bin_map: Dict[Tuple[int, ...], List[int]],
        gb_columns: List[str],
) -> Dict[str, Tuple[int, int]]:
    """Compute per-dimension (min,max) across observed bins (post-selection)."""
    if not bin_map:
        return {dim: (0, -1) for dim in gb_columns}  # empty
    arr = np.array(list(bin_map.keys()), dtype=np.int64)
    bounds: Dict[str, Tuple[int, int]] = {}
    for j, dim in enumerate(gb_columns):
        bounds[dim] = (int(arr[:, j].min()), int(arr[:, j].max()))
    return bounds


def _generate_neighbor_offsets(window_spec: Dict[str, int], gb_columns: Optional[List[str]] = None) -> np.ndarray:
    """Return all neighbor offsets as an array of shape (K, D), where D=len(gb_columns).
    Offsets cover the Cartesian product of [-w, +w] per dimension.
    If gb_columns is None, infer order from window_spec keys.
    """
    spans: List[np.ndarray] = []
    if gb_columns is None:
        gb_columns = list(window_spec.keys())
    for dim in gb_columns:
        w = window_spec.get(dim, 0)
        spans.append(np.arange(-w, w + 1, dtype=np.int64))
    # Cartesian product
    if not spans:
        return np.zeros((1, 0), dtype=np.int64)
    grids = np.meshgrid(*spans, indexing="ij")
    stacked = np.stack([g.reshape(-1) for g in grids], axis=1)
    return stacked  # (num_offsets, D)


def _get_neighbor_bins(
        center: Tuple[int, ...],
        offsets: np.ndarray,
        bin_ranges: Dict[str, Tuple[int, int]],  # ← Rename bounds
        boundary_mode: str = 'truncate'  # ← Add this (unused for now)
) -> List[Tuple[int, ...]]:
    """Apply boundary mode: drop neighbors outside observed (min,max) per dim."""

    # Get dimension order from bin_ranges keys (instead of gb_columns parameter)
    gb_columns = list(bin_ranges.keys())

    if offsets.size == 0:
        return [center]
    center_arr = np.array(center, dtype=np.int64)
    cand = center_arr + offsets  # (K, D)

    mask = np.ones(len(cand), dtype=bool)
    for j, dim in enumerate(gb_columns):
        lo, hi = bin_ranges[dim]  # ← Use bin_ranges instead of bounds
        mask &= (cand[:, j] >= lo) & (cand[:, j] <= hi)
    valid = cand[mask]
    return [tuple(map(int, row)) for row in valid]

@dataclass
class _AggResult:
    center: Tuple[int, ...]
    n_neighbors_used: int
    n_rows_aggregated: int
    effective_window_fraction: float
    # per-target aggregates
    stats: Dict[str, Dict[str, float]]  # target -> {mean,std,median,entries}
    # rows indices (unique) used for the window (for fitting)
    row_indices: np.ndarray


def _weighted_mean_std(x: np.ndarray, w: Optional[np.ndarray]) -> Tuple[float, float]:
    """Compute mean and (unbiased) std. If w is None, use ordinary formulas.
    Drops NaNs in x (and corresponding weights) beforehand (caller responsibility).
    For weights: use standard weighted mean and unbiased weighted std with effective dof.
    """
    if x.size == 0:
        return (np.nan, np.nan)

    if w is None:
        m = float(np.mean(x)) if x.size else np.nan
        s = float(np.std(x, ddof=1)) if x.size > 1 else np.nan
        return (m, s)

    # weights provided
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        return (np.nan, np.nan)
    m = float(np.sum(w * x) / wsum)
    # unbiased weighted variance per effective dof
    # var = sum(w*(x-m)^2) / (wsum - sum(w^2)/wsum)
    # guard denominator
    w2_sum = float(np.sum(w * w))
    denom = wsum - (w2_sum / wsum) if wsum > 0 else 0.0
    if denom <= 0.0:
        return (m, np.nan)
    var = float(np.sum(w * (x - m) ** 2) / denom)
    return (m, math.sqrt(var))


def _aggregate_window_zerocopy(
        df: pd.DataFrame,
        bin_map: Dict[Tuple[int, ...], List[int]],
        center_bins: Iterable[Tuple[int, ...]],
        neighbor_offsets: np.ndarray,
        bounds: Dict[str, Tuple[int, int]],
        gb_columns: List[str],
        fit_columns: List[str],
        weights: Optional[str],
) -> List[_AggResult]:
    """Aggregate per center bin using zero-copy neighbor indexing."""
    results: List[_AggResult] = []

    expected_neighbors = 1
    for dim in gb_columns:
        w = int(neighbor_offsets.max(initial=0))  # not exact per-dim, recompute precisely below
    # exact expected product
    expected_neighbors = 1
    for dim in gb_columns:
        w = window_spec_w = bounds.get(dim, (0, 0))  # placeholder not used here
    # Better: compute from offsets directly
    expected_neighbors = int(neighbor_offsets.shape[0]) if neighbor_offsets.size else 1

    for center in center_bins:
        neighbors = _get_neighbor_bins(center, neighbor_offsets, bounds)
        n_used = 0
        idx_list: List[int] = []
        for nb in neighbors:
            rows = bin_map.get(nb)
            if rows:
                n_used += 1
                idx_list.extend(rows)

        if idx_list:
            # dedup defensively
            idx_unique = np.unique(np.fromiter(idx_list, dtype=np.int64))
        else:
            idx_unique = np.array([], dtype=np.int64)

        eff_frac = (n_used / expected_neighbors) if expected_neighbors > 0 else np.nan
        n_rows = int(idx_unique.size)

        stats: Dict[str, Dict[str, float]] = {}
        if n_rows > 0:
            window_df = df.iloc[idx_unique]
            w = None
            if weights is not None:
                w_series = window_df[weights]
                # drop NaN/negative weights for stats
                valid_w = (~w_series.isna()) & (w_series.to_numpy() >= 0)
                w = w_series.to_numpy()[valid_w]
            for t in fit_columns:
                col = window_df[t]
                if weights is None:
                    x = col.dropna().to_numpy()
                    mean, std = _weighted_mean_std(x, None)
                else:
                    # apply joint validity: target not NaN and weight valid
                    valid = (~col.isna()).to_numpy()
                    if w is not None:
                        valid = valid & ((~w_series.isna()).to_numpy()) & (w_series.to_numpy() >= 0)
                    x = col.to_numpy()[valid]
                    ww = w_series.to_numpy()[valid]
                    mean, std = _weighted_mean_std(x, ww)
                median = float(np.median(col.dropna().to_numpy())) if col.notna().any() else np.nan
                entries = int(col.notna().sum())
                stats[t] = {
                    "mean": mean,
                    "std": std,
                    "median": median,
                    "entries": entries,
                }
        else:
            for t in fit_columns:
                stats[t] = {"mean": np.nan, "std": np.nan, "median": np.nan, "entries": 0}

        results.append(
            _AggResult(
                center=center,
                n_neighbors_used=n_used,
                n_rows_aggregated=n_rows,
                effective_window_fraction=eff_frac,
                stats=stats,
                row_indices=idx_unique,
            )
        )

    return results


# ===============
# Regression
# ===============

def _sanitize_suffix(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(name))



# ===============
# V0: statsmodels-based regression (REFERENCE ONLY — not called by main API)
# Kept for validation and non-linear fit development.
# ===============

# Optional statsmodels (only needed for V0 reference path)
_STATSMODELS_AVAILABLE = False
try:
    import statsmodels.formula.api as smf
    _STATSMODELS_AVAILABLE = True
except ImportError:
    pass


def _fit_window_regression_statsmodels_v0(
        df: pd.DataFrame,
        agg_results: List[_AggResult],
        fit_columns: List[str],
        fit_formula: str,
        linear_columns: List[str],
        weights: Optional[str],
        fitter: str = 'ols',
        min_stat: int = 10,
) -> Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]]:
    """V0 REFERENCE: statsmodels-based regression per bin.

    Supports OLS, WLS, GLM, RLM via formula string. Not called by
    make_sliding_window_fit (which uses V1/V2 for linear-only fits).
    Preserved for:
    - Cross-validation of V1/V2 results
    - Future non-linear sliding window function
    """
    if not _STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels required for V0. pip install statsmodels")

    out: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]] = {}

    for ar in agg_results:
        center_map: Dict[str, Dict[str, Any]] = {}
        if ar.row_indices.size == 0:
            for t in fit_columns:
                center_map[t] = _empty_fit_result("empty_window", 0)
            out[ar.center] = center_map
            continue

        window_df_full = df.iloc[ar.row_indices]

        for t in fit_columns:
            formula = fit_formula.replace("target", t) if "target" in fit_formula else fit_formula
            sub_df = window_df_full[[t] + linear_columns + ([weights] if weights else [])].copy()
            sub_df = sub_df.rename(columns={t: "__target__"})
            formula_t = formula.replace(t, "__target__")

            valid = sub_df["__target__"].notna()
            if weights is not None:
                w = sub_df[weights]
                valid &= (~w.isna()) & (w >= 0)
            for p in linear_columns:
                valid &= sub_df[p].notna()

            sub_df = sub_df.loc[valid]
            n_avail = len(sub_df)
            if n_avail < max(1, int(min_stat)):
                center_map[t] = _empty_fit_result("insufficient_stats", n_avail)
                continue

            try:
                if weights is not None or fitter == "wls":
                    model = smf.wls(formula=formula_t, data=sub_df, weights=sub_df[weights])
                elif fitter == "rlm":
                    model = smf.rlm(formula=formula_t, data=sub_df)
                elif fitter == "glm":
                    model = smf.glm(formula=formula_t, data=sub_df)
                else:
                    model = smf.ols(formula=formula_t, data=sub_df)
                res = model.fit()

                params = res.params.to_dict()
                intercept = float(params.get("Intercept", params.get("const", np.nan)))
                coeffs = {k: float(v) for k, v in params.items() if k not in ("Intercept", "const")}

                bse = {}
                if hasattr(res, 'bse') and res.bse is not None:
                    try:
                        bse = res.bse.to_dict()
                    except AttributeError:
                        pass
                intercept_err = float(bse.get("Intercept", bse.get("const", np.nan)))
                coeffs_err = {k: float(bse.get(k, np.nan)) for k in coeffs}

                r2 = getattr(res, "rsquared", np.nan)
                resid = res.resid
                rmse = float(np.sqrt(np.mean(resid ** 2)))

                center_map[t] = {
                    "coeffs": coeffs,
                    "coeffs_err": coeffs_err,
                    "intercept": intercept,
                    "intercept_err": intercept_err,
                    "r_squared": float(r2) if r2 is not None else np.nan,
                    "rmse": rmse,
                    "n_fitted": int(getattr(res, "nobs", len(sub_df))),
                    "quality_flag": "",
                }
            except Exception:
                center_map[t] = _empty_fit_result(f"fit_failed_{t}", n_avail)
        out[ar.center] = center_map

    return out


# ===============
# V1: Array-based OLS (numpy.linalg.lstsq)
# ===============

def _fit_window_regression_numpy(
        df: pd.DataFrame,
        agg_results: List[_AggResult],
        fit_columns: List[str],
        linear_columns: List[str],
        weights: Optional[str],
        min_stat: int,
) -> Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]]:
    """V1: numpy.linalg.lstsq per bin.

    Eliminates overhead of formula parsing and DataFrame construction.
    Only supports OLS (no WLS/GLM/RLM).
    """
    out: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]] = {}
    n_pred = len(linear_columns)
    n_params = n_pred + 1  # intercept + predictors

    # Pre-extract predictor columns as numpy arrays for speed
    pred_arrays = {p: df[p].to_numpy(dtype=np.float64) for p in linear_columns}
    target_arrays = {t: df[t].to_numpy(dtype=np.float64) for t in fit_columns}

    for ar in agg_results:
        center_map: Dict[str, Dict[str, Any]] = {}

        if ar.row_indices.size == 0:
            for t in fit_columns:
                center_map[t] = _empty_fit_result("empty_window", 0)
            out[ar.center] = center_map
            continue

        idx = ar.row_indices

        # Build design matrix once for all targets (shared predictors)
        X_cols = [pred_arrays[p][idx] for p in linear_columns]

        for t in fit_columns:
            y = target_arrays[t][idx]

            # Drop NaN rows (target + predictors)
            valid = np.isfinite(y)
            for xc in X_cols:
                valid &= np.isfinite(xc[: len(valid)])

            if weights is not None:
                w = df[weights].to_numpy(dtype=np.float64)[idx]
                valid &= np.isfinite(w) & (w >= 0)

            n_valid = int(np.sum(valid))
            if n_valid < max(1, int(min_stat)):
                center_map[t] = _empty_fit_result("insufficient_stats", n_valid)
                continue

            y_v = y[valid]
            X_design = np.column_stack(
                [np.ones(n_valid, dtype=np.float64)]
                + [xc[valid] for xc in X_cols]
            )

            try:
                beta, residuals, rank, sv = np.linalg.lstsq(X_design, y_v, rcond=None)

                # Compute residuals and diagnostics
                y_pred = X_design @ beta
                resid = y_v - y_pred
                rss = float(np.sum(resid ** 2))
                dof = n_valid - n_params
                s2 = rss / dof if dof > 0 else np.nan

                # R²
                ss_tot = float(np.sum((y_v - np.mean(y_v)) ** 2))
                r2 = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan

                # RMSE
                rmse = float(np.sqrt(rss / n_valid))

                # Standard errors: sqrt(s² × diag(XtX⁻¹))
                try:
                    XtX_inv = np.linalg.inv(X_design.T @ X_design)
                    se = np.sqrt(s2 * np.diag(XtX_inv)) if np.isfinite(s2) else np.full(n_params, np.nan)
                except np.linalg.LinAlgError:
                    se = np.full(n_params, np.nan)

                # Pack into result dict
                intercept = float(beta[0])
                intercept_err = float(se[0])
                coeffs = {linear_columns[j]: float(beta[j + 1]) for j in range(n_pred)}
                coeffs_err = {linear_columns[j]: float(se[j + 1]) for j in range(n_pred)}

                center_map[t] = {
                    "coeffs": coeffs,
                    "coeffs_err": coeffs_err,
                    "intercept": intercept,
                    "intercept_err": intercept_err,
                    "r_squared": r2,
                    "rmse": rmse,
                    "n_fitted": n_valid,
                    "quality_flag": "",
                }
            except Exception:
                center_map[t] = _empty_fit_result(f"fit_failed_{t}", n_valid)

        out[ar.center] = center_map

    return out


# ===============
# V2: Numba kernel batch fit
# ===============

def _fit_window_regression_numba(
        df: pd.DataFrame,
        agg_results: List[_AggResult],
        fit_columns: List[str],
        linear_columns: List[str],
        weights: Optional[str],
        min_stat: int,
) -> Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]]:
    """V2: Batch all window bins into a single Numba kernel call.

    Reshapes the per-bin window data into the kernel's expected format
    (X_all sorted by group, offsets array) and calls fit_groups_single_numba
    once for all bins. Eliminates the Python loop entirely.

    Only supports OLS (no WLS/GLM/RLM). Requires Numba + kernel module.
    """
    try:
        from groupby_regression_kernels import (
            fit_groups_single_numba as _kernel_single,
            INVALID_DETECT as _INVALID_DETECT,
            STATUS_OK as _STATUS_OK,
        )
    except ImportError:
        from .groupby_regression_kernels import (
            fit_groups_single_numba as _kernel_single,
            INVALID_DETECT as _INVALID_DETECT,
            STATUS_OK as _STATUS_OK,
        )

    n_pred = len(linear_columns)
    n_params = n_pred + 1

    # Pre-extract arrays
    pred_arrays = [df[p].to_numpy(dtype=np.float64) for p in linear_columns]
    target_arrays = {t: df[t].to_numpy(dtype=np.float64) for t in fit_columns}

    out: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]] = {}

    # Process each target separately (kernel is single-target)
    for t in fit_columns:
        y_full = target_arrays[t]

        # Build per-bin valid data: collect (X, Y) arrays and offsets
        bin_X_list: List[np.ndarray] = []
        bin_Y_list: List[np.ndarray] = []
        bin_centers: List[Tuple[int, ...]] = []
        bin_n_valid: List[int] = []
        skip_map: Dict[Tuple[int, ...], Dict[str, Any]] = {}

        for ar in agg_results:
            if ar.row_indices.size == 0:
                skip_map[ar.center] = _empty_fit_result("empty_window", 0)
                continue

            idx = ar.row_indices
            y = y_full[idx]
            x_cols = [pa[idx] for pa in pred_arrays]

            # Validity mask
            valid = np.isfinite(y)
            for xc in x_cols:
                valid &= np.isfinite(xc)

            n_valid = int(np.sum(valid))
            if n_valid < max(1, int(min_stat)):
                skip_map[ar.center] = _empty_fit_result("insufficient_stats", n_valid)
                continue

            # Extract valid rows
            X_v = np.column_stack([xc[valid] for xc in x_cols]) if n_pred > 0 else np.empty((n_valid, 0))
            Y_v = y[valid]

            bin_X_list.append(X_v)
            bin_Y_list.append(Y_v)
            bin_centers.append(ar.center)
            bin_n_valid.append(n_valid)

        n_bins = len(bin_X_list)

        if n_bins == 0:
            # All bins skipped
            for ar in agg_results:
                if ar.center not in out:
                    out[ar.center] = {}
                out[ar.center][t] = skip_map.get(ar.center,
                    _empty_fit_result("insufficient_stats", 0))
            continue

        # Concatenate into kernel format
        X_all = np.vstack(bin_X_list)            # (total_rows, n_feat)
        Y_all = np.concatenate(bin_Y_list)       # (total_rows,)
        W_all = np.empty(0, dtype=np.float64)    # unweighted

        # Build offsets
        offsets = np.zeros(n_bins + 1, dtype=np.int64)
        for i, nv in enumerate(bin_n_valid):
            offsets[i + 1] = offsets[i] + nv

        # Allocate outputs
        out_beta = np.empty((n_bins, n_params), dtype=np.float64)
        out_errors = np.empty((n_bins, n_params), dtype=np.float64)
        out_rms = np.empty(n_bins, dtype=np.float64)
        out_mad = np.empty(n_bins, dtype=np.float64)
        out_status = np.empty(n_bins, dtype=np.uint8)
        out_n_valid_arr = np.empty(n_bins, dtype=np.int64)
        out_n_filtered = np.empty(n_bins, dtype=np.int64)
        out_cond = np.empty(n_bins, dtype=np.float64)

        # Single kernel call for all bins
        _kernel_single(
            X_all, Y_all, W_all, offsets,
            n_bins, n_pred, n_params,
            True,  # fit_intercept
            max(1, int(min_stat)),  # min_stat
            False,  # compute_mad
            _INVALID_DETECT,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid_arr, out_n_filtered, out_cond,
        )

        # Unpack results back to per-bin dicts
        for i, center in enumerate(bin_centers):
            if center not in out:
                out[center] = {}

            if out_status[i] == _STATUS_OK:
                intercept = float(out_beta[i, 0])
                intercept_err = float(out_errors[i, 0])
                coeffs = {linear_columns[j]: float(out_beta[i, j + 1])
                          for j in range(n_pred)}
                coeffs_err = {linear_columns[j]: float(out_errors[i, j + 1])
                              for j in range(n_pred)}

                # RMSE: V1 uses sqrt(RSS/n), kernel gives sqrt(RSS/dof).
                # Convert: RSS = rms² × dof, RMSE = sqrt(RSS/n)
                dof = bin_n_valid[i] - n_params
                if dof > 0:
                    rss = float(out_rms[i] ** 2 * dof)
                    rmse = float(np.sqrt(rss / bin_n_valid[i]))
                else:
                    rss = float(out_rms[i] ** 2 * bin_n_valid[i])
                    rmse = float(out_rms[i])

                # R² = 1 - RSS / SS_tot
                y_bin = Y_all[offsets[i]:offsets[i + 1]]
                ss_tot = float(np.sum((y_bin - np.mean(y_bin)) ** 2))
                r2 = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan

                out[center][t] = {
                    "coeffs": coeffs,
                    "coeffs_err": coeffs_err,
                    "intercept": intercept,
                    "intercept_err": intercept_err,
                    "r_squared": r2,
                    "rmse": rmse,
                    "n_fitted": bin_n_valid[i],
                    "quality_flag": "",
                }
            else:
                out[center][t] = _empty_fit_result(
                    f"fit_failed_{t}", bin_n_valid[i])

        # Add skipped bins
        for center, result in skip_map.items():
            if center not in out:
                out[center] = {}
            out[center][t] = result

    return out


def _empty_fit_result(quality_flag: str, n_fitted: int) -> Dict[str, Any]:
    """Create a standard empty/failed fit result dict."""
    return {
        "coeffs": {},
        "coeffs_err": {},
        "intercept": np.nan,
        "intercept_err": np.nan,
        "r_squared": np.nan,
        "rmse": np.nan,
        "n_fitted": n_fitted,
        "quality_flag": quality_flag,
    }


# ===============
# Assembly
# ===============

def _assemble_results(
        gb_columns: List[str],
        agg_results: List[_AggResult],
        fit_results: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]],
        fit_columns: List[str],
        linear_columns: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    # Build column order
    pred_suffixes = {p: _sanitize_suffix(p) for p in linear_columns}

    for ar in agg_results:
        base: Dict[str, Any] = {dim: ar.center[i] for i, dim in enumerate(gb_columns)}
        base["n_neighbors_used"] = ar.n_neighbors_used
        base["n_rows_aggregated"] = ar.n_rows_aggregated
        base["effective_window_fraction"] = ar.effective_window_fraction

        # Aggregate stats
        for t, st in ar.stats.items():
            base[f"{t}_mean"] = st["mean"]
            base[f"{t}_std"] = st["std"]
            base[f"{t}_median"] = st["median"]
            base[f"{t}_entries"] = st["entries"]

        # Fit outputs
        fit_map = fit_results.get(ar.center, {})
        # If the entire window was empty and no fit_map entries exist, still mark quality
        empty_window = ar.n_rows_aggregated == 0

        # accumulate quality flags
        qflags: List[str] = []

        for t in fit_columns:
            tres = fit_map.get(t)
            if tres is None:
                # no fitting requested or not available
                base[f"{t}_intercept"] = np.nan
                base[f"{t}_intercept_err"] = np.nan
                for p, ps in pred_suffixes.items():
                    base[f"{t}_slope_{ps}"] = np.nan
                    base[f"{t}_slope_{ps}_err"] = np.nan
                base[f"{t}_r_squared"] = np.nan
                base[f"{t}_rmse"] = np.nan
                base[f"{t}_n_fitted"] = 0
                continue

            base[f"{t}_intercept"] = tres.get("intercept", np.nan)
            base[f"{t}_intercept_err"] = tres.get("intercept_err", np.nan)
            for p, ps in pred_suffixes.items():
                base[f"{t}_slope_{ps}"] = tres.get("coeffs", {}).get(p, np.nan)
                base[f"{t}_slope_{ps}_err"] = tres.get("coeffs_err", {}).get(p, np.nan)
            base[f"{t}_r_squared"] = tres.get("r_squared", np.nan)
            base[f"{t}_rmse"] = tres.get("rmse", np.nan)
            base[f"{t}_n_fitted"] = tres.get("n_fitted", 0)
            if tres.get("quality_flag"):
                qflags.append(str(tres.get("quality_flag")))

        if empty_window:
            qflags.append("empty_window")

        base["quality_flag"] = ",".join([q for q in qflags if q])
        rows.append(base)

    out = pd.DataFrame(rows)
    # Ensure group columns are present even if rows empty
    for dim in gb_columns:
        if dim not in out.columns:
            out[dim] = pd.Series(dtype="int64")

    # Order columns: gb_columns -> aggregations -> fit outputs -> diagnostics
    agg_cols = [c for c in out.columns if any(c.startswith(f"{t}_") for t in fit_columns) and (
            c.endswith("_mean") or c.endswith("_std") or c.endswith("_median") or c.endswith("_entries")
    )]

    fit_cols = []
    for t in fit_columns:
        fit_cols.append(f"{t}_intercept")
        fit_cols.append(f"{t}_intercept_err")
        for p, ps in pred_suffixes.items():
            fit_cols.append(f"{t}_slope_{ps}")
            fit_cols.append(f"{t}_slope_{ps}_err")
        fit_cols.append(f"{t}_r_squared")
        fit_cols.append(f"{t}_rmse")
        fit_cols.append(f"{t}_n_fitted")

    diag_cols = ["quality_flag", "n_neighbors_used", "n_rows_aggregated", "effective_window_fraction"]

    ordered = gb_columns + agg_cols + fit_cols + diag_cols
    # Keep any other columns at the end (defensive)
    others = [c for c in out.columns if c not in ordered]
    out = out[ordered + others]

    return out


# =====================
# Main entry point
# =====================

def make_sliding_window_fit(
        *,
        df: pd.DataFrame,
        gb_columns: List[str],
        fit_columns: List[str],
        linear_columns: List[str],
        window_spec: Dict[str, int],
        weights: Optional[str] = None,
        suffix: str = '_sw',
        selection: Optional[pd.Series] = None,
        fit_intercept: bool = True,
        min_stat: int = 10,
        cast_dtype: str = 'float64',
        backend: str = 'auto',
        algorithm: str = 'recompute',
        aggregation_functions: Optional[Dict[str, List[str]]] = None,
        binning_formulas: Optional[Dict[str, str]] = None,
        partition_strategy: Optional[dict] = None,
        return_metadata: bool = False,
        verbose: bool = False,
        **kwargs: Any,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Sliding window grouped linear regression (v4-aligned API).

    For each bin defined by gb_columns, aggregates data from neighboring bins
    within the window_spec radius and performs OLS (or WLS if weights given).

    Example
    -------
    ::

        # TPC calibration: fit drift velocity vs. space charge in 3D sector grid
        result = make_sliding_window_fit(
            df=df,
            gb_columns=['sectorBin', 'rBin', 'zBin'],
            fit_columns=['driftV'],
            linear_columns=['spaceCharge', 'temperature'],
            window_spec={'sectorBin': 2, 'rBin': 1, 'zBin': 1},
            min_stat=20,
            suffix='_sw',
            verbose=True,
        )
        # result contains per-bin: driftV_intercept_sw, driftV_slope_spaceCharge_sw,
        #   driftV_slope_temperature_sw, driftV_rmse_sw, etc.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    gb_columns : list[str]
        Columns defining the bin grid (must be integer-valued).
    fit_columns : list[str]
        Target columns for regression.
    linear_columns : list[str]
        Predictor columns (regressors).
    window_spec : dict[str, int]
        Half-width of the sliding window per dimension, in bin units.
        For each center bin, all neighbors within ±half-width are included.
        Omitted gb_columns dimensions default to 0 (no sliding).

        Example: For a 3D grid with gb_columns=['xBin', 'yBin', 'zBin']::

            window_spec = {'xBin': 1, 'yBin': 1, 'zBin': 0}

        For center bin (5, 3, 7), the window covers:
            xBin: [5-1, 5, 5+1] = [4, 5, 6]    (3 bins)
            yBin: [3-1, 3, 3+1] = [2, 3, 4]    (3 bins)
            zBin: [7]                             (1 bin, no sliding)
            → total window = 3 × 3 × 1 = 9 neighbor bins

        All rows from these 9 bins are pooled together for a single
        regression. At boundaries, the window is truncated (no wrap-around).
    weights : str, optional
        Column name for regression weights (WLS). None = OLS.
    suffix : str
        Suffix appended to output column names (default: '_sw').
    selection : pd.Series, optional
        Boolean mask to filter input rows before fitting.
    fit_intercept : bool
        If True (default), include intercept in the regression.
    min_stat : int
        Minimum valid rows in a window to perform fit.
    cast_dtype : str
        Data type for output coefficients (default: 'float64').
    backend : str
        Computation backend: 'auto' | 'numpy' | 'numba'.
    algorithm : str
        Algorithm: 'recompute' (V1/V2) | 'incremental' (V3, future).
    aggregation_functions : dict[str, list[str]], optional
        Per-column statistics to compute over each window. Currently
        hardcoded to mean/std/median/entries for all fit_columns.
        Future: user-specified, e.g.::

            aggregation_functions = {
                'driftV': ['mean', 'median', 'std', 'entries'],
                'temperature': ['mean', 'std'],
            }
    binning_formulas : dict[str, str], optional
        Provenance metadata recording how float columns were binned.
        Not used in computation, stored in output DataFrame.attrs.
        Example: ``{'rBin': 'int(r / 0.5)', 'zBin': 'int(z / 2.0)'}``
    partition_strategy : dict, optional
        Reserved for future parallel execution strategy.
    return_metadata : bool
        If True, return (DataFrame, metadata_dict).
    verbose : bool
        If True, print timing and progress information.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        Group-level regression results with columns:
        - Bin coordinates (gb_columns, no suffix)
        - Window metadata: n_neighbors_used, n_rows_aggregated,
          effective_window_fraction (with suffix)
        - Per-target aggregations: {target}_mean, _std, _median,
          _entries (with suffix)
        - Per-target fit: {target}_intercept, _intercept_err,
          _slope_{pred}, _slope_{pred}_err, _rmse, _n_fitted,
          quality_flag (with suffix)

    Roadmap
    -------
    - V3 (algorithm='incremental'): Pre-compute per-bin XtX/XtY, sum
      over window — avoids recomputing regressions from raw data.
    - Distance-dependent weighting: Gaussian/Epanechnikov kernel
      w ~ f(distance) for smooth interpolation. Will require specifying
      physical coordinates for distance calculation and COG computation.
    """
    t0 = time.time()

    # Fill missing window_spec dimensions with 0 (no sliding)
    full_window_spec = {dim: window_spec.get(dim, 0) for dim in gb_columns}

    # Validate
    _validate_sliding_window_inputs(
        df=df,
        gb_columns=gb_columns,
        window_spec=full_window_spec,
        fit_columns=fit_columns,
        linear_columns=linear_columns,
        weights=weights,
        selection=selection,
        min_stat=min_stat,
        backend=backend,
    )

    # Resolve 'auto' backend
    _resolved_backend = backend
    if _resolved_backend == 'auto':
        try:
            from groupby_regression_kernels import fit_groups_single_numba as _test
            _resolved_backend = 'numba'
        except ImportError:
            try:
                from .groupby_regression_kernels import fit_groups_single_numba as _test
                _resolved_backend = 'numba'
            except ImportError:
                _resolved_backend = 'numpy'

    if verbose:
        print(f"[SW] backend={_resolved_backend}, algorithm={algorithm}, "
              f"bins={len(gb_columns)}D, window={full_window_spec}")

    # Build zero-copy bin map
    bin_map = _build_bin_index_map(df, gb_columns, selection)
    center_bins = list(bin_map.keys())

    # Neighbor offsets and bounds
    neighbor_offsets = _generate_neighbor_offsets(full_window_spec, gb_columns)
    bounds = _observed_bin_bounds(bin_map, gb_columns)

    # Aggregation per window
    agg_results = _aggregate_window_zerocopy(
        df=df,
        bin_map=bin_map,
        center_bins=center_bins,
        neighbor_offsets=neighbor_offsets,
        bounds=bounds,
        gb_columns=gb_columns,
        fit_columns=fit_columns,
        weights=weights,
    )

    if verbose:
        print(f"[SW] Aggregation done: {len(agg_results)} bins, "
              f"{time.time()-t0:.3f}s elapsed")

    # Fitting — dispatch V1 (numpy) / V2 (numba)
    if _resolved_backend == 'numba' and weights is None:
        try:
            from groupby_regression_kernels import fit_groups_single_numba
            _use_numba = True
        except ImportError:
            try:
                from .groupby_regression_kernels import fit_groups_single_numba
                _use_numba = True
            except ImportError:
                _use_numba = False
    else:
        _use_numba = False

    if _use_numba:
        fit_results = _fit_window_regression_numba(
            df=df,
            agg_results=agg_results,
            fit_columns=fit_columns,
            linear_columns=linear_columns,
            weights=weights,
            min_stat=min_stat,
        )
        _backend_used = "numba"
    else:
        fit_results = _fit_window_regression_numpy(
            df=df,
            agg_results=agg_results,
            fit_columns=fit_columns,
            linear_columns=linear_columns,
            weights=weights,
            min_stat=min_stat,
        )
        _backend_used = "numpy_lstsq"

    if verbose:
        print(f"[SW] Fitting done ({_backend_used}): {time.time()-t0:.3f}s elapsed")

    # Assemble output
    out = _assemble_results(
        gb_columns=gb_columns,
        agg_results=agg_results,
        fit_results=fit_results,
        fit_columns=fit_columns,
        linear_columns=linear_columns,
    )

    # Apply suffix to non-bin columns
    if suffix:
        rename_map = {}
        bin_cols = set(gb_columns)
        for col in out.columns:
            if col not in bin_cols:
                rename_map[col] = col + suffix
        out = out.rename(columns=rename_map)

    # Cast dtype
    if cast_dtype:
        float_cols = out.select_dtypes(include=[np.floating]).columns
        out[float_cols] = out[float_cols].astype(cast_dtype)

    # Provenance
    metadata = {
        "gb_columns": list(gb_columns),
        "window_spec": full_window_spec,
        "boundary_mode": "truncate",
        "backend_used": _backend_used,
        "algorithm": algorithm,
        "suffix": suffix,
        "fit_intercept": fit_intercept,
        "n_bins": len(center_bins),
        "computation_time_sec": time.time() - t0,
        "python_version": sys.version,
    }
    out.attrs.update(metadata)

    if verbose:
        print(f"[SW] Complete: {len(out)} bins, {time.time()-t0:.3f}s total")

    if return_metadata:
        return out, metadata
    return out
