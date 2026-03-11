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


# ===============
# Kernel functions (V3b)
# ===============

def _resolve_kernel_width(
        kernel_width: Optional[Union[float, Dict[str, float]]],
        window_spec: Dict[str, int],
        gb_columns: List[str],
) -> Dict[str, float]:
    """Resolve kernel_width to per-dimension dict.

    If None, defaults to window_spec half-widths (σ = w means ~68% of
    Gaussian mass within the window). If float, same for all dims.
    """
    if kernel_width is None:
        return {dim: float(max(window_spec.get(dim, 0), 1)) for dim in gb_columns}
    if isinstance(kernel_width, (int, float)):
        return {dim: float(kernel_width) for dim in gb_columns}
    # dict — fill missing dims with window_spec defaults
    return {dim: float(kernel_width.get(dim, max(window_spec.get(dim, 0), 1)))
            for dim in gb_columns}


def _kernel_gaussian(offset: np.ndarray, sigma: np.ndarray) -> float:
    """Gaussian kernel: exp(-0.5 * ||offset/sigma||²)."""
    scaled = offset / sigma
    return float(np.exp(-0.5 * np.sum(scaled ** 2)))


def _kernel_epanechnikov(offset: np.ndarray, sigma: np.ndarray) -> float:
    """Epanechnikov kernel: max(0, 1 - ||offset/sigma||²)."""
    u2 = float(np.sum((offset / sigma) ** 2))
    return max(0.0, 1.0 - u2)


def _kernel_linear(offset: np.ndarray, sigma: np.ndarray) -> float:
    """Linear decay kernel: max(0, 1 - ||offset/sigma||)."""
    u = float(np.sqrt(np.sum((offset / sigma) ** 2)))
    return max(0.0, 1.0 - u)


def _kernel_uniform(offset: np.ndarray, sigma: np.ndarray) -> float:
    """Uniform kernel: w = 1 for all neighbors."""
    return 1.0


_KERNEL_REGISTRY: Dict[str, Callable] = {
    'uniform': _kernel_uniform,
    'gaussian': _kernel_gaussian,
    'epanechnikov': _kernel_epanechnikov,
    'linear': _kernel_linear,
}


def _precompute_offset_weights(
        offsets: np.ndarray,
        kernel: Union[str, Callable],
        kernel_width_vec: np.ndarray,
) -> np.ndarray:
    """Compute weight for each offset in the offset table.

    Returns array of shape (K,) with non-negative weights.
    Weights depend only on offset (not on center), so they are
    precomputable once for the entire grid.
    """
    if isinstance(kernel, str):
        kernel_fn = _KERNEL_REGISTRY.get(kernel)
        if kernel_fn is None:
            raise ValueError(
                f"Unknown kernel '{kernel}'. "
                f"Available: {list(_KERNEL_REGISTRY.keys())}"
            )
    else:
        kernel_fn = kernel

    K = offsets.shape[0]
    weights = np.empty(K, dtype=np.float64)
    for i in range(K):
        weights[i] = kernel_fn(offsets[i].astype(np.float64), kernel_width_vec)
    return weights


# ===============
# Boundary handling (V3b)
# ===============

def _resolve_boundary(
        boundary: Union[str, Dict[str, str]],
        gb_columns: List[str],
) -> Dict[str, str]:
    """Resolve boundary to per-dimension dict."""
    valid_modes = ('full', 'symmetric', 'periodic')
    if isinstance(boundary, str):
        if boundary not in valid_modes:
            raise ValueError(f"boundary must be one of {valid_modes}, got '{boundary}'")
        return {dim: boundary for dim in gb_columns}
    for dim, mode in boundary.items():
        if mode not in valid_modes:
            raise ValueError(f"boundary['{dim}'] must be one of {valid_modes}, got '{mode}'")
    # Fill missing dims with 'full' (default)
    return {dim: boundary.get(dim, 'full') for dim in gb_columns}


def _validate_periodic_dims(
        boundary_resolved: Dict[str, str],
        bounds: Dict[str, Tuple[int, int]],
        window_spec: Dict[str, int],
) -> None:
    """Validate periodic dimensions have enough bins (P1-4, P1-7)."""
    for dim, mode in boundary_resolved.items():
        if mode == 'periodic':
            lo, hi = bounds[dim]
            n_bins = hi - lo + 1
            w = window_spec.get(dim, 0)
            if n_bins < 2 * w + 1:
                raise ValueError(
                    f"Periodic dimension '{dim}' has {n_bins} bins but "
                    f"window={w} requires at least {2*w+1}. "
                    f"Reduce window or use more bins."
                )


def _get_neighbor_bins_v2(
        center: Tuple[int, ...],
        offsets: np.ndarray,
        bin_ranges: Dict[str, Tuple[int, int]],
        boundary_resolved: Dict[str, str],
        window_spec: Dict[str, int],
) -> Tuple[List[Tuple[int, ...]], np.ndarray]:
    """Boundary-aware neighbor generation (V3b).

    Returns (neighbor_bins, valid_offset_indices) — the indices into the
    offsets array for the surviving neighbors, needed to look up
    precomputed weights.

    Boundary modes per dimension:
    - 'full': truncate at observed range (current behavior)
    - 'symmetric': limit to max symmetric extent around center
    - 'periodic': wrap around [lo, hi] range
    """
    gb_columns = list(bin_ranges.keys())

    if offsets.size == 0:
        return [center], np.array([0], dtype=np.int64)

    center_arr = np.array(center, dtype=np.int64)
    D = len(gb_columns)

    # For symmetric mode: compute effective window per dimension
    effective_offsets = offsets.copy()
    mask = np.ones(len(offsets), dtype=bool)

    for j, dim in enumerate(gb_columns):
        lo, hi = bin_ranges[dim]
        mode = boundary_resolved[dim]
        w = window_spec.get(dim, 0)

        if mode == 'symmetric':
            # Max symmetric extent: min distance to either boundary
            max_left = center_arr[j] - lo
            max_right = hi - center_arr[j]
            eff_w = min(w, max_left, max_right)
            # Filter offsets to [-eff_w, +eff_w] in this dimension
            mask &= (offsets[:, j] >= -eff_w) & (offsets[:, j] <= eff_w)

        elif mode == 'periodic':
            # Wrap: candidate = ((center + offset - lo) % range) + lo
            n_range = hi - lo + 1
            raw = center_arr[j] + offsets[:, j]
            wrapped = ((raw - lo) % n_range) + lo
            effective_offsets[:, j] = wrapped - center_arr[j]
            # All periodic neighbors are valid (no truncation)

        else:  # 'full'
            cand = center_arr[j] + offsets[:, j]
            mask &= (cand >= lo) & (cand <= hi)

    # Apply mask
    valid_indices = np.where(mask)[0]
    valid_offsets = effective_offsets[valid_indices]

    # Compute actual neighbor coordinates
    neighbors = center_arr + valid_offsets
    result = [tuple(map(int, row)) for row in neighbors]

    return result, valid_indices


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
    # per-agg_column aggregates (COG etc.)
    agg_stats: Optional[Dict[str, Dict[str, float]]] = None  # col -> {mean,std,median}


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
        agg_columns: Optional[List[str]] = None,
        agg_median: bool = False,
) -> List[_AggResult]:
    """Aggregate per center bin using zero-copy neighbor indexing.

    Pre-extracts numpy arrays from DataFrame once, then uses direct
    array indexing per bin — no Pandas operations in the hot loop.

    Parameters
    ----------
    agg_columns : list[str], optional
        Additional columns to aggregate (mean, std, and optionally median)
        within each window. Useful for computing center-of-gravity of
        groupby variables or predictor columns.
    agg_median : bool
        If True, also compute (unweighted) median for agg_columns.
    """
    results: List[_AggResult] = []
    expected_neighbors = int(neighbor_offsets.shape[0]) if neighbor_offsets.size else 1

    # Pre-extract numpy arrays ONCE — eliminates all Pandas overhead in the loop
    target_arrays = {t: df[t].to_numpy(dtype=np.float64) for t in fit_columns}
    w_array = df[weights].to_numpy(dtype=np.float64) if weights is not None else None

    # Pre-extract agg_columns arrays
    _agg_cols = agg_columns or []
    agg_arrays = {c: df[c].to_numpy(dtype=np.float64) for c in _agg_cols}

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
            idx_unique = np.unique(np.fromiter(idx_list, dtype=np.int64))
        else:
            idx_unique = np.array([], dtype=np.int64)

        eff_frac = (n_used / expected_neighbors) if expected_neighbors > 0 else np.nan
        n_rows = int(idx_unique.size)

        stats: Dict[str, Dict[str, float]] = {}
        agg_st: Optional[Dict[str, Dict[str, float]]] = None

        if n_rows > 0:
            # Weight validity mask (computed once per window, shared across targets)
            if w_array is not None:
                w_win = w_array[idx_unique]
                w_valid = np.isfinite(w_win) & (w_win >= 0)
            else:
                w_win = None
                w_valid = None

            for t in fit_columns:
                y = target_arrays[t][idx_unique]
                y_finite = np.isfinite(y)

                if weights is None:
                    x = y[y_finite]
                    mean, std = _weighted_mean_std(x, None)
                else:
                    valid = y_finite & w_valid
                    x = y[valid]
                    ww = w_win[valid]
                    mean, std = _weighted_mean_std(x, ww)

                n_finite = int(np.sum(y_finite))
                if n_finite > 0:
                    median = float(np.median(y[y_finite]))
                else:
                    median = np.nan
                stats[t] = {
                    "mean": mean,
                    "std": std,
                    "median": median,
                    "entries": n_finite,
                }

            # Aggregate extra columns (COG etc.)
            if _agg_cols:
                agg_st = {}
                for c in _agg_cols:
                    y = agg_arrays[c][idx_unique]
                    y_finite = np.isfinite(y)

                    if weights is None:
                        x = y[y_finite]
                        mean, std = _weighted_mean_std(x, None)
                    else:
                        valid = y_finite & w_valid
                        x = y[valid]
                        ww = w_win[valid]
                        mean, std = _weighted_mean_std(x, ww)

                    if agg_median and int(np.sum(y_finite)) > 0:
                        median = float(np.median(y[y_finite]))
                    else:
                        median = np.nan

                    agg_st[c] = {"mean": mean, "std": std, "median": median}
        else:
            for t in fit_columns:
                stats[t] = {"mean": np.nan, "std": np.nan, "median": np.nan, "entries": 0}
            if _agg_cols:
                agg_st = {c: {"mean": np.nan, "std": np.nan, "median": np.nan} for c in _agg_cols}

        results.append(
            _AggResult(
                center=center,
                n_neighbors_used=n_used,
                n_rows_aggregated=n_rows,
                effective_window_fraction=eff_frac,
                stats=stats,
                row_indices=idx_unique,
                agg_stats=agg_st,
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
# V3: Pre-computed per-bin XtX/XtY, summed over window (incremental algorithm)
# ===============

@dataclass
class _BinSuffStats:
    """Sufficient statistics for a single bin, single target."""
    XtX: np.ndarray       # (p, p) — X^T X
    XtY: np.ndarray       # (p,)   — X^T y
    n: int                # number of valid rows
    sum_y: float          # Σ y_i  (for R²)
    sum_y2: float         # Σ y_i² (for R²)


def _precompute_bin_sufficient_stats(
        df: pd.DataFrame,
        bin_map: Dict[Tuple[int, ...], List[int]],
        fit_columns: List[str],
        linear_columns: List[str],
        fit_intercept: bool = True,
        agg_columns: Optional[List[str]] = None,
) -> Tuple[Dict[str, Dict[Tuple[int, ...], '_BinSuffStats']],
           Optional[Dict[str, Dict[Tuple[int, ...], Tuple[float, float, int]]]]]:
    """Pre-compute XtX, XtY, n, Σy, Σy² for each bin and target.

    If agg_columns is provided, also compute per-bin (sum, sum_sq, n) for
    each agg column — lightweight sufficient stats for mean/std.

    Returns
    -------
    (fit_stats, agg_stats) where:
      fit_stats: dict[target_name, dict[bin_key, _BinSuffStats]]
      agg_stats: dict[agg_col, dict[bin_key, (sum_y, sum_y2, n)]] or None
    """
    n_pred = len(linear_columns)
    n_params = n_pred + (1 if fit_intercept else 0)

    # Pre-extract full arrays
    pred_arrays = [df[p].to_numpy(dtype=np.float64) for p in linear_columns]
    target_arrays = {t: df[t].to_numpy(dtype=np.float64) for t in fit_columns}

    result: Dict[str, Dict[Tuple[int, ...], _BinSuffStats]] = {}

    for t in fit_columns:
        y_full = target_arrays[t]
        bin_stats: Dict[Tuple[int, ...], _BinSuffStats] = {}

        for bin_key, row_indices in bin_map.items():
            idx = np.array(row_indices, dtype=np.int64)
            y = y_full[idx]
            x_cols = [pa[idx] for pa in pred_arrays]

            # Validity mask: finite target + finite predictors
            valid = np.isfinite(y)
            for xc in x_cols:
                valid &= np.isfinite(xc)

            n_valid = int(np.sum(valid))
            if n_valid == 0:
                bin_stats[bin_key] = _BinSuffStats(
                    XtX=np.zeros((n_params, n_params), dtype=np.float64),
                    XtY=np.zeros(n_params, dtype=np.float64),
                    n=0,
                    sum_y=0.0,
                    sum_y2=0.0,
                )
                continue

            y_v = y[valid]

            # Build design matrix
            if fit_intercept:
                X = np.column_stack(
                    [np.ones(n_valid, dtype=np.float64)]
                    + [xc[valid] for xc in x_cols]
                )
            else:
                X = np.column_stack([xc[valid] for xc in x_cols])

            bin_stats[bin_key] = _BinSuffStats(
                XtX=X.T @ X,
                XtY=X.T @ y_v,
                n=n_valid,
                sum_y=float(np.sum(y_v)),
                sum_y2=float(np.sum(y_v ** 2)),
            )

        result[t] = bin_stats

    # Agg columns: lightweight sufficient stats (sum, sum_sq, n)
    agg_result: Optional[Dict[str, Dict[Tuple[int, ...], Tuple[float, float, int]]]] = None
    if agg_columns:
        agg_result = {}
        agg_arrays = {c: df[c].to_numpy(dtype=np.float64) for c in agg_columns}
        for c in agg_columns:
            a_full = agg_arrays[c]
            c_stats: Dict[Tuple[int, ...], Tuple[float, float, int]] = {}
            for bin_key, row_indices in bin_map.items():
                idx = np.array(row_indices, dtype=np.int64)
                y = a_full[idx]
                valid = np.isfinite(y)
                n_v = int(np.sum(valid))
                if n_v > 0:
                    y_v = y[valid]
                    c_stats[bin_key] = (float(np.sum(y_v)), float(np.sum(y_v ** 2)), n_v)
                else:
                    c_stats[bin_key] = (0.0, 0.0, 0)
            agg_result[c] = c_stats

    return result, agg_result


def _compute_lightweight_agg_results(
        bin_map: Dict[Tuple[int, ...], List[int]],
        center_bins: List[Tuple[int, ...]],
        neighbor_offsets: np.ndarray,
        bounds: Dict[str, Tuple[int, int]],
        gb_columns: List[str],
        fit_columns: List[str],
        bin_suff_stats: Dict[str, Dict[Tuple[int, ...], _BinSuffStats]],
        boundary_resolved: Optional[Dict[str, str]] = None,
        window_spec: Optional[Dict[str, int]] = None,
        offset_weights: Optional[np.ndarray] = None,
        use_weighted_kernel: bool = False,
        prebuilt_neighbor_table: Optional[Tuple] = None,
        agg_columns: Optional[List[str]] = None,
        agg_suff_stats: Optional[Dict[str, Dict[Tuple[int, ...], Tuple[float, float, int]]]] = None,
        agg_median: bool = False,
) -> List[_AggResult]:
    """Compute lightweight _AggResult entries from sufficient statistics.

    Derives mean/std/entries from summed sufficient stats over the window.
    Median is NaN (cannot compute from sufficient statistics).

    V3b extensions:
    - boundary_resolved: per-dim boundary mode for neighbor generation
    - offset_weights: precomputed kernel weight per offset
    - use_weighted_kernel: if True, use weighted formulas for stats (P1-2)
    - prebuilt_neighbor_table: (nbr_indices, nbr_weights, nbr_counts) to skip
      redundant _get_neighbor_bins_v2 calls — share with _build_neighbor_table
    """
    expected_neighbors = int(neighbor_offsets.shape[0]) if neighbor_offsets.size else 1
    use_v2_neighbors = (boundary_resolved is not None and window_spec is not None)

    # If we have a prebuilt table, use index-based lookup instead of per-bin _get_neighbor_bins_v2
    if prebuilt_neighbor_table is not None:
        nbr_indices, nbr_weights_tbl, nbr_counts = prebuilt_neighbor_table
        bin_idx_map = {b: i for i, b in enumerate(center_bins)}
        use_prebuilt = True
    else:
        use_prebuilt = False

    results: List[_AggResult] = []
    for ci, center in enumerate(center_bins):
        if use_prebuilt:
            # Use prebuilt neighbor table — no _get_neighbor_bins_v2 call
            nc = int(nbr_counts[ci])
            neighbor_bin_indices = nbr_indices[ci, :nc]
            nbr_weights_local = nbr_weights_tbl[ci, :nc] if use_weighted_kernel else None

            n_used = 0
            for k in range(nc):
                j = int(neighbor_bin_indices[k])
                if j >= 0 and center_bins[j] in bin_map and len(bin_map[center_bins[j]]) > 0:
                    n_used += 1

            stats: Dict[str, Dict[str, float]] = {}
            n_rows_total = 0

            for t in fit_columns:
                target_stats = bin_suff_stats[t]

                if use_weighted_kernel and nbr_weights_local is not None:
                    w_sum_y = 0.0
                    w_sum_y2 = 0.0
                    w_n = 0.0
                    n_raw = 0
                    for k in range(nc):
                        j = int(neighbor_bin_indices[k])
                        if j < 0:
                            continue
                        nb = center_bins[j]
                        bs = target_stats.get(nb)
                        if bs is not None and bs.n > 0:
                            w = float(nbr_weights_local[k])
                            w_sum_y += w * bs.sum_y
                            w_sum_y2 += w * bs.sum_y2
                            w_n += w * bs.n
                            n_raw += bs.n

                    if w_n > 0:
                        mean = w_sum_y / w_n
                        var = w_sum_y2 / w_n - mean ** 2
                        std = float(np.sqrt(var)) if var > 0 else 0.0
                        stats[t] = {"mean": mean, "std": std, "median": np.nan, "entries": n_raw}
                    else:
                        stats[t] = {"mean": np.nan, "std": np.nan, "median": np.nan, "entries": 0}
                    n_rows_total = max(n_rows_total, n_raw)
                else:
                    sum_y = 0.0
                    sum_y2 = 0.0
                    n_valid = 0
                    for k in range(nc):
                        j = int(neighbor_bin_indices[k])
                        if j < 0:
                            continue
                        nb = center_bins[j]
                        bs = target_stats.get(nb)
                        if bs is not None and bs.n > 0:
                            n_valid += bs.n
                            sum_y += bs.sum_y
                            sum_y2 += bs.sum_y2

                    if n_valid > 0:
                        mean = sum_y / n_valid
                        var_num = sum_y2 - n_valid * mean ** 2
                        std = float(np.sqrt(var_num / (n_valid - 1))) if n_valid > 1 and var_num > 0 else np.nan
                        stats[t] = {"mean": mean, "std": std, "median": np.nan, "entries": n_valid}
                    else:
                        stats[t] = {"mean": np.nan, "std": np.nan, "median": np.nan, "entries": 0}
                    n_rows_total = max(n_rows_total, n_valid)

            # Compute agg_columns stats from sufficient stats
            agg_st = None
            _agg_cols = agg_columns or []
            if _agg_cols and agg_suff_stats is not None:
                agg_st = {}
                for c in _agg_cols:
                    c_stats = agg_suff_stats.get(c, {})
                    if use_weighted_kernel and nbr_weights_local is not None:
                        w_sy, w_sy2, w_n = 0.0, 0.0, 0.0
                        for k in range(nc):
                            j = int(neighbor_bin_indices[k])
                            if j < 0:
                                continue
                            nb = center_bins[j]
                            cs = c_stats.get(nb)
                            if cs is not None and cs[2] > 0:
                                w = float(nbr_weights_local[k])
                                w_sy += w * cs[0]
                                w_sy2 += w * cs[1]
                                w_n += w * cs[2]
                        if w_n > 0:
                            mean = w_sy / w_n
                            var = w_sy2 / w_n - mean ** 2
                            std = float(np.sqrt(var)) if var > 0 else 0.0
                        else:
                            mean, std = np.nan, np.nan
                    else:
                        sy, sy2, nv = 0.0, 0.0, 0
                        for k in range(nc):
                            j = int(neighbor_bin_indices[k])
                            if j < 0:
                                continue
                            nb = center_bins[j]
                            cs = c_stats.get(nb)
                            if cs is not None and cs[2] > 0:
                                sy += cs[0]; sy2 += cs[1]; nv += cs[2]
                        if nv > 0:
                            mean = sy / nv
                            var_num = sy2 - nv * mean ** 2
                            std = float(np.sqrt(var_num / (nv - 1))) if nv > 1 and var_num > 0 else np.nan
                        else:
                            mean, std = np.nan, np.nan
                    agg_st[c] = {"mean": mean, "std": std, "median": np.nan}

            eff_frac = (n_used / expected_neighbors) if expected_neighbors > 0 else np.nan
            results.append(_AggResult(
                center=center, n_neighbors_used=n_used,
                n_rows_aggregated=n_rows_total, effective_window_fraction=eff_frac,
                stats=stats, row_indices=np.array([], dtype=np.int64),
                agg_stats=agg_st,
            ))
            continue

        # Original path (no prebuilt table)
        if use_v2_neighbors:
            neighbors, valid_idx = _get_neighbor_bins_v2(
                center, neighbor_offsets, bounds, boundary_resolved, window_spec)
            nbr_weights = offset_weights[valid_idx] if offset_weights is not None else None
        else:
            neighbors = _get_neighbor_bins(center, neighbor_offsets, bounds)
            nbr_weights = None

        # Count used neighbors and total rows
        n_used = 0
        for nb in neighbors:
            if nb in bin_map and len(bin_map[nb]) > 0:
                n_used += 1

        # Compute per-target stats from sufficient statistics
        stats: Dict[str, Dict[str, float]] = {}
        n_rows_total = 0

        for t in fit_columns:
            target_stats = bin_suff_stats[t]

            if use_weighted_kernel and nbr_weights is not None:
                # Weighted stats (P1-2: population-weighted, no Bessel)
                w_sum_y = 0.0
                w_sum_y2 = 0.0
                w_n = 0.0
                n_raw = 0
                for k, nb in enumerate(neighbors):
                    bs = target_stats.get(nb)
                    if bs is not None and bs.n > 0:
                        w = nbr_weights[k]
                        w_sum_y += w * bs.sum_y
                        w_sum_y2 += w * bs.sum_y2
                        w_n += w * bs.n
                        n_raw += bs.n

                if w_n > 0:
                    mean = w_sum_y / w_n
                    # Population-weighted variance (P1-2: no Bessel)
                    var = w_sum_y2 / w_n - mean ** 2
                    std = float(np.sqrt(var)) if var > 0 else 0.0
                    stats[t] = {
                        "mean": mean,
                        "std": std,
                        "median": np.nan,
                        "entries": n_raw,  # unweighted integer count
                    }
                else:
                    stats[t] = {"mean": np.nan, "std": np.nan, "median": np.nan, "entries": 0}
                n_rows_total = max(n_rows_total, n_raw)
            else:
                # Unweighted stats (original V3 path)
                sum_y = 0.0
                sum_y2 = 0.0
                n_valid = 0
                for nb in neighbors:
                    bs = target_stats.get(nb)
                    if bs is not None and bs.n > 0:
                        n_valid += bs.n
                        sum_y += bs.sum_y
                        sum_y2 += bs.sum_y2

                if n_valid > 0:
                    mean = sum_y / n_valid
                    var_num = sum_y2 - n_valid * mean ** 2
                    std = float(np.sqrt(var_num / (n_valid - 1))) if n_valid > 1 and var_num > 0 else np.nan
                    stats[t] = {
                        "mean": mean,
                        "std": std,
                        "median": np.nan,
                        "entries": n_valid,
                    }
                else:
                    stats[t] = {"mean": np.nan, "std": np.nan, "median": np.nan, "entries": 0}
                n_rows_total = max(n_rows_total, n_valid)

        eff_frac = (n_used / expected_neighbors) if expected_neighbors > 0 else np.nan

        # Compute agg_columns stats from sufficient stats (original path)
        agg_st = None
        _agg_cols = agg_columns or []
        if _agg_cols and agg_suff_stats is not None:
            agg_st = {}
            for c in _agg_cols:
                c_stats = agg_suff_stats.get(c, {})
                if use_weighted_kernel and nbr_weights is not None:
                    w_sy, w_sy2, w_n = 0.0, 0.0, 0.0
                    for k, nb in enumerate(neighbors):
                        cs = c_stats.get(nb)
                        if cs is not None and cs[2] > 0:
                            w = nbr_weights[k]
                            w_sy += w * cs[0]
                            w_sy2 += w * cs[1]
                            w_n += w * cs[2]
                    if w_n > 0:
                        mean = w_sy / w_n
                        var = w_sy2 / w_n - mean ** 2
                        std = float(np.sqrt(var)) if var > 0 else 0.0
                    else:
                        mean, std = np.nan, np.nan
                else:
                    sy, sy2, nv = 0.0, 0.0, 0
                    for nb in neighbors:
                        cs = c_stats.get(nb)
                        if cs is not None and cs[2] > 0:
                            sy += cs[0]; sy2 += cs[1]; nv += cs[2]
                    if nv > 0:
                        mean = sy / nv
                        var_num = sy2 - nv * mean ** 2
                        std = float(np.sqrt(var_num / (nv - 1))) if nv > 1 and var_num > 0 else np.nan
                    else:
                        mean, std = np.nan, np.nan
                agg_st[c] = {"mean": mean, "std": std, "median": np.nan}

        results.append(_AggResult(
            center=center,
            n_neighbors_used=n_used,
            n_rows_aggregated=n_rows_total,
            effective_window_fraction=eff_frac,
            stats=stats,
            row_indices=np.array([], dtype=np.int64),
            agg_stats=agg_st,
        ))

    return results


def _fit_window_regression_incremental(
        bin_map: Dict[Tuple[int, ...], List[int]],
        bin_suff_stats: Dict[str, Dict[Tuple[int, ...], _BinSuffStats]],
        center_bins: List[Tuple[int, ...]],
        neighbor_offsets: np.ndarray,
        bounds: Dict[str, Tuple[int, int]],
        fit_columns: List[str],
        linear_columns: List[str],
        fit_intercept: bool,
        min_stat: int,
        agg_results: List[_AggResult],
        boundary_resolved: Optional[Dict[str, str]] = None,
        window_spec: Optional[Dict[str, int]] = None,
        offset_weights: Optional[np.ndarray] = None,
        use_weighted_kernel: bool = False,
        prebuilt_neighbor_table: Optional[Tuple] = None,
) -> Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]]:
    """V3/V3b: Solve normal equations from summed per-bin sufficient statistics.

    For each center bin c and target t:
        XtX_w = Σ_{b ∈ N(c)} w_b · XtX_bin[b]
        XtY_w = Σ_{b ∈ N(c)} w_b · XtY_bin[b]
        beta  = solve(XtX_w, XtY_w)

    V3b extensions:
    - boundary_resolved: per-dim boundary mode for _get_neighbor_bins_v2
    - offset_weights: precomputed kernel weight per offset index
    - use_weighted_kernel: if True, _err columns → NaN (P1-3)
    - prebuilt_neighbor_table: (nbr_indices, nbr_weights, nbr_counts) to skip
      redundant _get_neighbor_bins_v2 calls
    """
    n_pred = len(linear_columns)
    n_params = n_pred + (1 if fit_intercept else 0)
    use_v2_neighbors = (boundary_resolved is not None and window_spec is not None)

    # If we have a prebuilt table, use index-based lookup
    if prebuilt_neighbor_table is not None:
        nbr_indices_tbl, nbr_weights_tbl, nbr_counts_tbl = prebuilt_neighbor_table
        use_prebuilt = True
    else:
        use_prebuilt = False

    out: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]] = {}

    for ci, center in enumerate(center_bins):
        if use_prebuilt:
            nc = int(nbr_counts_tbl[ci])
            # Build neighbors list from prebuilt table
            neighbors = []
            nbr_weights = np.empty(nc, dtype=np.float64) if use_weighted_kernel else None
            for k in range(nc):
                j = int(nbr_indices_tbl[ci, k])
                if j >= 0:
                    neighbors.append(center_bins[j])
                    if nbr_weights is not None:
                        nbr_weights[k] = nbr_weights_tbl[ci, k]
        elif use_v2_neighbors:
            neighbors, valid_idx = _get_neighbor_bins_v2(
                center, neighbor_offsets, bounds, boundary_resolved, window_spec)
            nbr_weights = offset_weights[valid_idx] if offset_weights is not None else None
        else:
            neighbors = _get_neighbor_bins(center, neighbor_offsets, bounds)
            nbr_weights = None

        center_map: Dict[str, Dict[str, Any]] = {}

        for t in fit_columns:
            target_stats = bin_suff_stats[t]

            # Sum (weighted) sufficient statistics over window neighbors
            XtX_w = np.zeros((n_params, n_params), dtype=np.float64)
            XtY_w = np.zeros(n_params, dtype=np.float64)
            n_total = 0       # unweighted row count (P1-1: for min_stat)
            sum_y_total = 0.0
            sum_y2_total = 0.0

            for k, nb in enumerate(neighbors):
                bs = target_stats.get(nb)
                if bs is not None and bs.n > 0:
                    w_b = nbr_weights[k] if nbr_weights is not None else 1.0
                    XtX_w += w_b * bs.XtX
                    XtY_w += w_b * bs.XtY
                    n_total += bs.n          # unweighted (P1-1)
                    sum_y_total += w_b * bs.sum_y
                    sum_y2_total += w_b * bs.sum_y2

            # min_stat uses unweighted count (P1-1)
            if n_total < max(1, int(min_stat)):
                center_map[t] = _empty_fit_result("insufficient_stats", n_total)
                continue

            # Solve normal equations: XtX_w @ beta = XtY_w
            try:
                beta = np.linalg.solve(XtX_w, XtY_w)
            except np.linalg.LinAlgError:
                center_map[t] = _empty_fit_result("singular_matrix", n_total)
                continue

            # Diagnostics from sufficient stats:
            # RSS = Σ w·y² - β' · (Σ w·X'y) = sum_y2_total - beta @ XtY_w
            rss = sum_y2_total - float(beta @ XtY_w)
            if rss < 0:
                rss = 0.0

            dof = n_total - n_params
            s2 = rss / dof if dof > 0 else np.nan

            # RMSE = sqrt(RSS / n)
            rmse = float(np.sqrt(rss / n_total)) if n_total > 0 else np.nan

            # R²: use weighted sums for consistency
            # SS_tot = Σ w·y² - (Σ w·y)² / (Σ w·n)
            # But we need Σ w·n for weighted mean calculation
            if nbr_weights is not None:
                w_n_total = 0.0
                for k, nb in enumerate(neighbors):
                    bs = target_stats.get(nb)
                    if bs is not None and bs.n > 0:
                        w_n_total += nbr_weights[k] * bs.n
                y_mean = sum_y_total / w_n_total if w_n_total > 0 else 0.0
                ss_tot = sum_y2_total - w_n_total * y_mean ** 2
            else:
                y_mean = sum_y_total / n_total if n_total > 0 else 0.0
                ss_tot = sum_y2_total - n_total * y_mean ** 2
            r2 = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan

            # Standard errors (P1-3): NaN when kernel != 'uniform'
            if use_weighted_kernel:
                se = np.full(n_params, np.nan)
            else:
                try:
                    XtX_inv = np.linalg.inv(XtX_w)
                    se = np.sqrt(s2 * np.diag(XtX_inv)) if np.isfinite(s2) else np.full(n_params, np.nan)
                except np.linalg.LinAlgError:
                    se = np.full(n_params, np.nan)

            # Pack result
            if fit_intercept:
                intercept = float(beta[0])
                intercept_err = float(se[0])
                coeffs = {linear_columns[j]: float(beta[j + 1]) for j in range(n_pred)}
                coeffs_err = {linear_columns[j]: float(se[j + 1]) for j in range(n_pred)}
            else:
                intercept = 0.0
                intercept_err = 0.0
                coeffs = {linear_columns[j]: float(beta[j]) for j in range(n_pred)}
                coeffs_err = {linear_columns[j]: float(se[j]) for j in range(n_pred)}

            center_map[t] = {
                "coeffs": coeffs,
                "coeffs_err": coeffs_err,
                "intercept": intercept,
                "intercept_err": intercept_err,
                "r_squared": r2,
                "rmse": rmse,
                "n_fitted": n_total,
                "quality_flag": "",
            }

        out[center] = center_map

    return out


# ===============
# V3-Numba: Incremental solve kernel
# ===============

def _check_numba_available() -> bool:
    """Check if Numba is importable."""
    try:
        import numba
        return True
    except ImportError:
        return False


def _flatten_bins_for_v4(
        df,
        gb_columns: List[str],
        fit_columns: List[str],
        linear_columns: List[str],
        selection=None,
        fit_intercept: bool = True,
):
    """Convert DataFrame + bin structure to flat integer arrays for V4 Numba kernels.

    Returns
    -------
    bin_ids : ndarray[n_rows] — flat bin index per row (0..n_bins-1), -1 if filtered
    X_all : ndarray[n_rows, n_linear] — predictor values
    Y_all : ndarray[n_rows, n_targets] — target values
    n_bins : int — number of unique bins
    center_bins : list[tuple] — bin keys ordered by flat index
    bin_coords : ndarray[n_bins, n_dims] — integer bin coordinates
    bounds : dict[str, (int, int)] — per-dim (min, max)
    """
    if selection is not None:
        sel_mask = selection.to_numpy().astype(bool)
    else:
        sel_mask = np.ones(len(df), dtype=bool)

    gb_arrays = [df[c].to_numpy(dtype=np.int64) for c in gb_columns]
    n_dims = len(gb_columns)
    n_rows = len(df)

    selected_rows = np.flatnonzero(sel_mask)
    if len(selected_rows) == 0:
        return (np.full(n_rows, -1, dtype=np.int64),
                np.empty((n_rows, len(linear_columns)), dtype=np.float64),
                np.empty((n_rows, len(fit_columns)), dtype=np.float64),
                0, [], np.empty((0, n_dims), dtype=np.int64), {})

    coords_selected = np.column_stack([a[selected_rows] for a in gb_arrays])
    unique_coords, inverse = np.unique(coords_selected, axis=0, return_inverse=True)
    n_bins = unique_coords.shape[0]

    bin_ids = np.full(n_rows, -1, dtype=np.int64)
    bin_ids[selected_rows] = inverse

    center_bins = [tuple(int(x) for x in row) for row in unique_coords]

    bounds = {}
    for j, dim in enumerate(gb_columns):
        bounds[dim] = (int(unique_coords[:, j].min()), int(unique_coords[:, j].max()))

    X_all = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in linear_columns]) if linear_columns else np.empty((n_rows, 0), dtype=np.float64)
    Y_all = np.column_stack([df[c].to_numpy(dtype=np.float64) for c in fit_columns])

    return bin_ids, X_all, Y_all, n_bins, center_bins, unique_coords, bounds


def _build_neighbor_table_vectorized(
        bin_coords: np.ndarray,
        neighbor_offsets: np.ndarray,
        bounds: Dict[str, Tuple[int, int]],
        gb_columns: List[str],
        boundary_resolved: Dict[str, str],
        window_spec: Dict[str, int],
        offset_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build neighbor table using vectorized numpy — no Python loop over bins.

    Works per-dimension to avoid large (n_bins, n_offsets, n_dims) allocation.
    """
    n_bins = bin_coords.shape[0]
    n_dims = bin_coords.shape[1]
    n_offsets = neighbor_offsets.shape[0]

    bounds_arr = np.array([(bounds[d][0], bounds[d][1]) for d in gb_columns], dtype=np.int64)
    mins = bounds_arr[:, 0]
    maxs = bounds_arr[:, 1]
    sizes = maxs - mins + 1

    grid = np.full(tuple(int(s) for s in sizes), -1, dtype=np.int64)
    for i in range(n_bins):
        idx = tuple(int(bin_coords[i, d] - mins[d]) for d in range(n_dims))
        grid[idx] = i

    nbr_valid = np.ones((n_bins, n_offsets), dtype=bool)
    clipped_per_dim = []

    for d, dim in enumerate(gb_columns):
        bmode = boundary_resolved.get(dim, 'full')
        lo, hi = int(mins[d]), int(maxs[d])
        dim_size = hi - lo + 1
        w = window_spec.get(dim, 0)

        nbr_d = bin_coords[:, d:d+1] + neighbor_offsets[np.newaxis, :, d]

        if bmode == 'periodic':
            nbr_d = ((nbr_d - lo) % dim_size) + lo
        elif bmode == 'symmetric':
            center_d = bin_coords[:, d:d+1]
            offset_d = np.abs(neighbor_offsets[:, d])
            max_left = center_d - lo
            max_right = hi - center_d
            eff_w = np.minimum(w, np.minimum(max_left, max_right))
            nbr_valid &= (offset_d[np.newaxis, :] <= eff_w)

        out_of_range = (nbr_d < lo) | (nbr_d > hi)
        nbr_valid &= ~out_of_range
        clipped_per_dim.append(np.clip(nbr_d - lo, 0, dim_size - 1).astype(np.intp))

    nbr_flat = grid[tuple(clipped_per_dim)]
    nbr_flat[~nbr_valid] = -1

    is_valid = (nbr_flat >= 0)
    nbr_counts = is_valid.sum(axis=1).astype(np.int64)
    max_nbr = int(nbr_counts.max()) if n_bins > 0 else 0

    if max_nbr == n_offsets and int(nbr_counts.min()) == n_offsets:
        return nbr_flat.copy(), np.broadcast_to(offset_weights, (n_bins, n_offsets)).copy(), nbr_counts

    all_valid_mask = (nbr_counts == n_offsets)
    nbr_indices = np.full((n_bins, max_nbr), -1, dtype=np.int64)
    nbr_weights_out = np.zeros((n_bins, max_nbr), dtype=np.float64)

    n_interior = int(all_valid_mask.sum())
    if n_interior > 0:
        interior_idx = np.flatnonzero(all_valid_mask)
        nbr_indices[interior_idx, :] = nbr_flat[interior_idx, :max_nbr]
        nbr_weights_out[interior_idx, :] = offset_weights[np.newaxis, :max_nbr]

    edge_idx = np.flatnonzero(~all_valid_mask)
    for ei in range(len(edge_idx)):
        i = edge_idx[ei]
        nc = int(nbr_counts[i])
        if nc > 0:
            valid_idx = np.flatnonzero(is_valid[i])[:nc]
            nbr_indices[i, :nc] = nbr_flat[i, valid_idx]
            nbr_weights_out[i, :nc] = offset_weights[valid_idx]

    return nbr_indices, nbr_weights_out, nbr_counts


_NUMBA_V4_KERNEL_CACHE = None  # Module-level cache: avoids 470ms/call overhead of re-defining closures

def _get_numba_v4_kernels():
    """JIT-compile V4 Loop 1 (accumulate) and Loop 2 (solve) kernels.

    Cached at module level after first call to avoid the ~470ms Python overhead
    of defining nested @nb.njit closures on every invocation.
    """
    global _NUMBA_V4_KERNEL_CACHE
    if _NUMBA_V4_KERNEL_CACHE is not None:
        return _NUMBA_V4_KERNEL_CACHE

    import numba as nb

    @nb.njit(cache=True)
    def _solve_cholesky_inplace(A, b, p):
        """Cholesky solve A @ x = b in place. Returns success flag."""
        for i in range(p):
            for j in range(i):
                s = 0.0
                for kk in range(j):
                    s += A[i, kk] * A[j, kk]
                A[i, j] = (A[i, j] - s) / A[j, j]
            s = 0.0
            for kk in range(i):
                s += A[i, kk] * A[i, kk]
            val = A[i, i] - s
            if val <= 0.0:
                return False
            A[i, i] = math.sqrt(val)
        for i in range(p):
            s = 0.0
            for kk in range(i):
                s += A[i, kk] * b[kk]
            b[i] = (b[i] - s) / A[i, i]
        for i in range(p - 1, -1, -1):
            s = 0.0
            for kk in range(i + 1, p):
                s += A[kk, i] * b[kk]
            b[i] = (b[i] - s) / A[i, i]
        return True

    @nb.njit(cache=True)
    def _cholesky_diag_inv(L, p):
        """Compute diag(A^-1) from Cholesky factor L."""
        diag = np.empty(p, dtype=np.float64)
        e = np.zeros(p, dtype=np.float64)
        for col in range(p):
            for i in range(p):
                e[i] = 0.0
            e[col] = 1.0
            for i in range(p):
                s = 0.0
                for kk in range(i):
                    s += L[i, kk] * e[kk]
                e[i] = (e[i] - s) / L[i, i]
            for i in range(p - 1, -1, -1):
                s = 0.0
                for kk in range(i + 1, p):
                    s += L[kk, i] * e[kk]
                e[i] = (e[i] - s) / L[i, i]
            diag[col] = e[col]
        return diag

    @nb.njit(cache=True)
    def accumulate_bin_stats(
            bin_ids, X_all, Y_all, n_bins, n_params, n_targets, fit_intercept,
            XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all,
    ):
        """Loop 1: Stream raw data, accumulate per-bin XtX/XtY."""
        n_rows = bin_ids.shape[0]
        n_linear = X_all.shape[1]

        for row in range(n_rows):
            b = bin_ids[row]
            if b < 0:
                continue
            x_ok = True
            for j in range(n_linear):
                if not np.isfinite(X_all[row, j]):
                    x_ok = False
                    break
            if not x_ok:
                continue
            x = np.empty(n_params, dtype=np.float64)
            if fit_intercept:
                x[0] = 1.0
                for j in range(n_linear):
                    x[j + 1] = X_all[row, j]
            else:
                for j in range(n_linear):
                    x[j] = X_all[row, j]
            for t in range(n_targets):
                y = Y_all[row, t]
                if not np.isfinite(y):
                    continue
                n_all[t, b] += 1
                sum_y_all[t, b] += y
                sum_y2_all[t, b] += y * y
                for p in range(n_params):
                    XtY_all[t, b, p] += x[p] * y
                    for q in range(p + 1):
                        XtX_all[t, b, p, q] += x[p] * x[q]
        # Symmetrize XtX
        for t in range(n_targets):
            for b in range(n_bins):
                for p in range(n_params):
                    for q in range(p + 1, n_params):
                        XtX_all[t, b, p, q] = XtX_all[t, b, q, p]

    @nb.njit(cache=True)
    def solve_all_windows(
            XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all,
            nbr_indices, nbr_weights, nbr_counts,
            n_params, min_stat, use_weighted_kernel,
            beta_out, se_out, rmse_out, r2_out, n_fitted_out, status_out,
            mean_out, std_out, entries_out,
    ):
        """Loop 2: For each center bin, sum neighbors + solve + compute stats."""
        n_bins = XtX_all.shape[0]
        p = n_params

        for i in range(n_bins):
            nc = nbr_counts[i]
            XtX_w = np.zeros((p, p), dtype=np.float64)
            XtY_w = np.zeros(p, dtype=np.float64)
            n_total = 0
            sum_y = 0.0
            sum_y2 = 0.0
            w_n = 0.0

            for k in range(nc):
                j = nbr_indices[i, k]
                if j < 0 or n_all[j] == 0:
                    continue
                w = nbr_weights[i, k]
                n_j = n_all[j]
                n_total += n_j
                sum_y += w * sum_y_all[j]
                sum_y2 += w * sum_y2_all[j]
                w_n += w * n_j
                for r in range(p):
                    XtY_w[r] += w * XtY_all[j, r]
                    for c in range(p):
                        XtX_w[r, c] += w * XtX_all[j, r, c]

            n_fitted_out[i] = n_total
            entries_out[i] = n_total

            if n_total > 0:
                if use_weighted_kernel and w_n > 0.0:
                    mean_val = sum_y / w_n
                    var_val = sum_y2 / w_n - mean_val * mean_val
                    mean_out[i] = mean_val
                    std_out[i] = math.sqrt(max(0.0, var_val))
                else:
                    mean_val = sum_y / n_total
                    mean_out[i] = mean_val
                    # ddof=1 (Bessel correction) to match np.std(data, ddof=1)
                    if n_total > 1:
                        var_val = (sum_y2 - n_total * mean_val * mean_val) / (n_total - 1)
                        std_out[i] = math.sqrt(max(0.0, var_val))
                    else:
                        std_out[i] = np.nan
            else:
                mean_out[i] = np.nan
                std_out[i] = np.nan

            if n_total < min_stat:
                status_out[i] = 1
                for r in range(p):
                    beta_out[i, r] = np.nan
                    se_out[i, r] = np.nan
                rmse_out[i] = np.nan
                r2_out[i] = np.nan
                continue

            A = XtX_w.copy()
            b = XtY_w.copy()
            ok = _solve_cholesky_inplace(A, b, p)

            if not ok:
                status_out[i] = 2
                for r in range(p):
                    beta_out[i, r] = np.nan
                    se_out[i, r] = np.nan
                rmse_out[i] = np.nan
                r2_out[i] = np.nan
                continue

            for r in range(p):
                beta_out[i, r] = b[r]

            rss = sum_y2
            for r in range(p):
                rss -= b[r] * XtY_w[r]
            if rss < 0.0:
                rss = 0.0

            rmse_out[i] = math.sqrt(rss / n_total) if n_total > 0 else np.nan

            if w_n > 0.0:
                y_mean = sum_y / w_n
            else:
                y_mean = 0.0
            ss_tot = sum_y2 - w_n * y_mean * y_mean
            if ss_tot > 0.0:
                r2_out[i] = 1.0 - rss / ss_tot
            else:
                r2_out[i] = np.nan

            if use_weighted_kernel:
                for r in range(p):
                    se_out[i, r] = np.nan
            else:
                dof = n_total - p
                if dof > 0:
                    s2 = rss / dof
                    diag_inv = _cholesky_diag_inv(A, p)
                    for r in range(p):
                        if diag_inv[r] > 0.0:
                            se_out[i, r] = math.sqrt(s2 * diag_inv[r])
                        else:
                            se_out[i, r] = np.nan
                else:
                    for r in range(p):
                        se_out[i, r] = np.nan

            status_out[i] = 0

    _NUMBA_V4_KERNEL_CACHE = (accumulate_bin_stats, solve_all_windows)
    return _NUMBA_V4_KERNEL_CACHE


def _fit_incremental_v4_numba(
        df,
        gb_columns: List[str],
        fit_columns: List[str],
        linear_columns: List[str],
        window_spec: Dict[str, int],
        neighbor_offsets: np.ndarray,
        offset_weights: np.ndarray,
        boundary_resolved: Dict[str, str],
        fit_intercept: bool = True,
        min_stat: int = 5,
        selection=None,
        kernel: str = 'uniform',
        verbose: bool = False,
) -> Tuple[Dict, List[_AggResult]]:
    """V4 Numba sliding window: flatten → accumulate → solve, all in Numba.

    Replaces V3-numba. Returns (fit_results, agg_results) compatible with
    _assemble_results.
    """
    t0 = time.time()

    n_linear = len(linear_columns)
    n_targets = len(fit_columns)
    n_params = n_linear + (1 if fit_intercept else 0)
    _is_weighted_kernel = (kernel != 'uniform') if isinstance(kernel, str) else True

    # Step 1: Flatten DataFrame to arrays
    bin_ids, X_all, Y_all, n_bins, center_bins, bin_coords, bounds = \
        _flatten_bins_for_v4(df, gb_columns, fit_columns, linear_columns, selection, fit_intercept)

    if n_bins == 0:
        return {}, []

    if verbose:
        print(f"[V4] Flatten: {n_bins} bins, {X_all.shape[0]} rows, {time.time()-t0:.3f}s")

    # Step 2: Build vectorized neighbor table
    nbr_indices, nbr_weights, nbr_counts = _build_neighbor_table_vectorized(
        bin_coords, neighbor_offsets, bounds, gb_columns,
        boundary_resolved, window_spec, offset_weights)

    if verbose:
        print(f"[V4] Neighbor table: {time.time()-t0:.3f}s")

    # Step 3: JIT kernels
    accumulate_bin_stats, solve_all_windows = _get_numba_v4_kernels()

    # Step 4: Allocate + Loop 1 (accumulate)
    XtX_all = np.zeros((n_targets, n_bins, n_params, n_params), dtype=np.float64)
    XtY_all = np.zeros((n_targets, n_bins, n_params), dtype=np.float64)
    n_all = np.zeros((n_targets, n_bins), dtype=np.int64)
    sum_y_all = np.zeros((n_targets, n_bins), dtype=np.float64)
    sum_y2_all = np.zeros((n_targets, n_bins), dtype=np.float64)

    t_l1 = time.time()
    accumulate_bin_stats(
        bin_ids, X_all, Y_all, n_bins, n_params, n_targets, fit_intercept,
        XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all)
    t_l1 = time.time() - t_l1

    if verbose:
        print(f"[V4] Loop 1 (accumulate): {t_l1:.4f}s  [cum: {time.time()-t0:.3f}s]")

    # Step 5: Loop 2 (solve) per target
    all_beta = np.full((n_targets, n_bins, n_params), np.nan, dtype=np.float64)
    all_se = np.full((n_targets, n_bins, n_params), np.nan, dtype=np.float64)
    all_rmse = np.full((n_targets, n_bins), np.nan, dtype=np.float64)
    all_r2 = np.full((n_targets, n_bins), np.nan, dtype=np.float64)
    all_n_fitted = np.zeros((n_targets, n_bins), dtype=np.int64)
    all_status = np.zeros((n_targets, n_bins), dtype=np.int64)
    all_mean = np.full((n_targets, n_bins), np.nan, dtype=np.float64)
    all_std = np.full((n_targets, n_bins), np.nan, dtype=np.float64)
    all_entries = np.zeros((n_targets, n_bins), dtype=np.int64)

    t_l2 = time.time()
    for ti in range(n_targets):
        solve_all_windows(
            XtX_all[ti], XtY_all[ti], n_all[ti], sum_y_all[ti], sum_y2_all[ti],
            nbr_indices, nbr_weights, nbr_counts,
            n_params, min_stat, _is_weighted_kernel,
            all_beta[ti], all_se[ti], all_rmse[ti], all_r2[ti],
            all_n_fitted[ti], all_status[ti],
            all_mean[ti], all_std[ti], all_entries[ti])
    t_l2 = time.time() - t_l2

    if verbose:
        print(f"[V4] Loop 2 (solve):  {t_l2:.4f}s  [cum: {time.time()-t0:.3f}s]")

    # Step 6: Unpack to dict format for _assemble_results
    n_pred = len(linear_columns)
    fit_results: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]] = {}
    for i, center in enumerate(center_bins):
        center_map: Dict[str, Dict[str, Any]] = {}
        for ti, t in enumerate(fit_columns):
            status = int(all_status[ti, i])
            if status != 0:
                reason = "insufficient_stats" if status == 1 else "singular_matrix"
                center_map[t] = _empty_fit_result(reason, int(all_n_fitted[ti, i]))
                continue

            beta = all_beta[ti, i]
            se = all_se[ti, i]

            if fit_intercept:
                intercept = float(beta[0])
                intercept_err = float(se[0])
                coeffs = {linear_columns[j]: float(beta[j + 1]) for j in range(n_pred)}
                coeffs_err = {linear_columns[j]: float(se[j + 1]) for j in range(n_pred)}
            else:
                intercept = 0.0
                intercept_err = 0.0
                coeffs = {linear_columns[j]: float(beta[j]) for j in range(n_pred)}
                coeffs_err = {linear_columns[j]: float(se[j]) for j in range(n_pred)}

            center_map[t] = {
                "coeffs": coeffs,
                "coeffs_err": coeffs_err,
                "intercept": intercept,
                "intercept_err": intercept_err,
                "r_squared": float(all_r2[ti, i]),
                "rmse": float(all_rmse[ti, i]),
                "n_fitted": int(all_n_fitted[ti, i]),
                "quality_flag": "",
            }
        fit_results[center] = center_map

    # Build compatible agg_results
    expected_neighbors = int(neighbor_offsets.shape[0]) if neighbor_offsets.size else 1
    agg_results: List[_AggResult] = []
    for i, center in enumerate(center_bins):
        nc = int(nbr_counts[i])
        stats: Dict[str, Dict[str, float]] = {}
        for ti, t in enumerate(fit_columns):
            stats[t] = {
                "mean": float(all_mean[ti, i]),
                "std": float(all_std[ti, i]),
                "median": np.nan,  # Cannot compute from sufficient stats
                "entries": int(all_entries[ti, i]),
            }
        agg_results.append(_AggResult(
            center=center,
            n_neighbors_used=nc,
            n_rows_aggregated=int(all_entries[0, i]) if n_targets > 0 else 0,
            effective_window_fraction=nc / expected_neighbors if expected_neighbors > 0 else np.nan,
            stats=stats,
            row_indices=np.array([], dtype=np.int64),
        ))

    if verbose:
        print(f"[V4] Unpack + total: {time.time()-t0:.3f}s")

    return fit_results, agg_results


# ###########################################################################
# V5: numpy-in / numpy-out — eliminates DataFrame conversion overhead
# ###########################################################################

def _flatten_bins_for_v5(
        df: pd.DataFrame,
        gb_columns: List[str],
        fit_columns: List[str],
        linear_columns: List[str],
        selection=None,
        fit_intercept: bool = True,
):
    """Fast DataFrame → flat arrays. Avoids np.unique axis=0 by using ravel_multi_index.

    Returns
    -------
    bin_ids : ndarray[n_rows] int64 — bin index per row (0..n_bins-1), -1 if filtered
    X_all : ndarray[n_rows, n_linear] float64
    Y_all : ndarray[n_rows, n_targets] float64
    n_bins : int
    bin_coords : ndarray[n_bins, n_dims] int64 — (row, col, ...) per bin
    bounds : dict[str, (int, int)]
    """
    n_rows = len(df)
    n_dims = len(gb_columns)

    # Extract group arrays — single pass over DataFrame
    gb_arrays = [df[c].to_numpy(dtype=np.int64) for c in gb_columns]

    # Selection mask
    if selection is not None:
        sel = selection.to_numpy().astype(bool) if hasattr(selection, 'to_numpy') else np.asarray(selection, dtype=bool)
    else:
        sel = None

    # Bounds per dimension
    bounds = {}
    mins = np.empty(n_dims, dtype=np.int64)
    maxs = np.empty(n_dims, dtype=np.int64)
    sizes = np.empty(n_dims, dtype=np.int64)
    for d, dim in enumerate(gb_columns):
        a = gb_arrays[d]
        if sel is not None:
            a_sel = a[sel]
        else:
            a_sel = a
        if len(a_sel) == 0:
            return (np.full(n_rows, -1, dtype=np.int64),
                    np.empty((n_rows, len(linear_columns)), dtype=np.float64),
                    np.empty((n_rows, len(fit_columns)), dtype=np.float64),
                    0, np.empty((0, n_dims), dtype=np.int64), {})
        lo, hi = int(a_sel.min()), int(a_sel.max())
        bounds[dim] = (lo, hi)
        mins[d] = lo
        maxs[d] = hi
        sizes[d] = hi - lo + 1

    # Ravel multi-index: (d0, d1, d2) → flat index in dense grid
    # Much faster than np.unique(coords, axis=0)
    shifted = [gb_arrays[d] - mins[d] for d in range(n_dims)]
    strides = np.ones(n_dims, dtype=np.int64)
    for d in range(n_dims - 2, -1, -1):
        strides[d] = strides[d + 1] * sizes[d + 1]

    flat_grid_ids = np.zeros(n_rows, dtype=np.int64)
    for d in range(n_dims):
        flat_grid_ids += shifted[d] * strides[d]

    # Apply selection
    if sel is not None:
        flat_grid_ids[~sel] = -1

    # Find occupied bins — which flat_grid_ids actually have data
    valid_mask = flat_grid_ids >= 0
    occupied = np.unique(flat_grid_ids[valid_mask])
    n_bins = len(occupied)

    # Map flat_grid_id → compact bin index (0..n_bins-1)
    grid_total = int(np.prod(sizes))
    remap = np.full(grid_total, -1, dtype=np.int64)
    remap[occupied] = np.arange(n_bins, dtype=np.int64)

    bin_ids = np.where(valid_mask, remap[flat_grid_ids], -1)

    # Reconstruct bin coordinates from flat index
    bin_coords = np.empty((n_bins, n_dims), dtype=np.int64)
    for d in range(n_dims):
        bin_coords[:, d] = (occupied // strides[d]) % sizes[d] + mins[d]

    # Extract predictor and target arrays — single pass
    if linear_columns:
        X_all = df[linear_columns].to_numpy(dtype=np.float64, copy=False)
    else:
        X_all = np.empty((n_rows, 0), dtype=np.float64)
    Y_all = df[fit_columns].to_numpy(dtype=np.float64, copy=False)

    return bin_ids, X_all, Y_all, n_bins, bin_coords, bounds


def make_sliding_window_fit_v5_arrays(
        *,
        bin_ids: np.ndarray,
        X_all: np.ndarray,
        Y_all: np.ndarray,
        n_bins: int,
        bin_coords: np.ndarray,
        bounds: Dict[str, Tuple[int, int]],
        gb_columns: List[str],
        fit_columns: List[str],
        linear_columns: List[str],
        window_spec: Dict[str, int],
        boundary: Union[str, Dict[str, str]] = 'full',
        kernel: Union[str, Callable] = 'uniform',
        kernel_width: Optional[Union[float, Dict[str, float]]] = None,
        fit_intercept: bool = True,
        min_stat: int = 10,
        verbose: bool = False,
        _collect_timings: bool = False,
) -> Dict[str, np.ndarray]:
    """V5 numpy-in / numpy-out sliding window regression.

    This is the hot-path function. No DataFrame conversion.
    Accepts pre-extracted arrays and returns flat numpy arrays.

    Parameters
    ----------
    bin_ids : ndarray[n_rows] int64 — bin index per row, -1 for excluded
    X_all : ndarray[n_rows, n_linear] float64 — predictor values
    Y_all : ndarray[n_rows, n_targets] float64 — target values
    n_bins : int — number of unique bins
    bin_coords : ndarray[n_bins, n_dims] int64 — bin grid coordinates
    bounds : dict[str, (min, max)] — per-dimension bounds
    gb_columns : list[str] — group-by column names
    fit_columns : list[str] — target column names
    linear_columns : list[str] — predictor column names
    window_spec : dict[str, int] — half-width per dimension
    boundary : str or dict — boundary mode ('full', 'symmetric', 'periodic')
    kernel : str — weighting kernel ('uniform', 'gaussian', etc.)
    kernel_width : optional — kernel width parameter
    fit_intercept : bool — include intercept
    min_stat : int — minimum rows for valid fit
    verbose : bool — print timing

    Returns
    -------
    dict of str → ndarray, all shape (n_bins,) or (n_bins, n_params):
        'bin_coords': ndarray[n_bins, n_dims] int64
        Per target t, predictor p:
            '{t}_intercept': ndarray[n_bins]
            '{t}_intercept_err': ndarray[n_bins]
            '{t}_slope_{p}': ndarray[n_bins]
            '{t}_slope_{p}_err': ndarray[n_bins]
            '{t}_r_squared': ndarray[n_bins]
            '{t}_rmse': ndarray[n_bins]
            '{t}_n_fitted': ndarray[n_bins] int64
            '{t}_mean': ndarray[n_bins]
            '{t}_std': ndarray[n_bins]
            '{t}_entries': ndarray[n_bins] int64
        'n_neighbors_used': ndarray[n_bins] int64
        'n_rows_aggregated': ndarray[n_bins] int64
        'effective_window_fraction': ndarray[n_bins] float64
        'status': ndarray[n_bins] int64  (0=ok, 1=insufficient, 2=singular)
    """
    t0 = time.time()

    n_linear = len(linear_columns)
    n_targets = len(fit_columns)
    n_params = n_linear + (1 if fit_intercept else 0)
    _is_weighted_kernel = (kernel != 'uniform') if isinstance(kernel, str) else True

    if n_bins == 0:
        return {'bin_coords': bin_coords}

    # --- Setup: offsets, boundary, kernel weights, array allocation ---
    t_setup_start = time.time()
    full_window_spec = {dim: window_spec.get(dim, 0) for dim in gb_columns}
    neighbor_offsets = _generate_neighbor_offsets(full_window_spec, gb_columns)
    boundary_resolved = _resolve_boundary(boundary, gb_columns)
    _validate_periodic_dims(boundary_resolved, bounds, full_window_spec)
    kernel_width_resolved = _resolve_kernel_width(kernel_width, full_window_spec, gb_columns)
    kernel_width_vec = np.array([kernel_width_resolved[dim] for dim in gb_columns], dtype=np.float64)
    offset_weights = _precompute_offset_weights(neighbor_offsets, kernel, kernel_width_vec)
    t_setup = time.time() - t_setup_start

    # --- Neighbor table ---
    t_nbr_start = time.time()
    nbr_indices, nbr_weights, nbr_counts = _build_neighbor_table_vectorized(
        bin_coords, neighbor_offsets, bounds, gb_columns,
        boundary_resolved, full_window_spec, offset_weights)
    t_nbr = time.time() - t_nbr_start

    if verbose:
        print(f"[V5] Neighbor table: {n_bins} bins, {time.time()-t0:.4f}s")

    # JIT kernels
    accumulate_bin_stats, solve_all_windows = _get_numba_v4_kernels()

    # --- Loop 1: accumulate per-bin XtX/XtY ---
    XtX_all = np.zeros((n_targets, n_bins, n_params, n_params), dtype=np.float64)
    XtY_all = np.zeros((n_targets, n_bins, n_params), dtype=np.float64)
    n_all = np.zeros((n_targets, n_bins), dtype=np.int64)
    sum_y_all = np.zeros((n_targets, n_bins), dtype=np.float64)
    sum_y2_all = np.zeros((n_targets, n_bins), dtype=np.float64)

    t_l1_start = time.time()
    accumulate_bin_stats(
        bin_ids, X_all, Y_all, n_bins, n_params, n_targets, fit_intercept,
        XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all)
    t_l1 = time.time() - t_l1_start

    if verbose:
        print(f"[V5] Loop 1 (accumulate): {t_l1:.4f}s  [cum: {time.time()-t0:.3f}s]")

    # --- Loop 2: solve per window ---
    all_beta = np.full((n_targets, n_bins, n_params), np.nan, dtype=np.float64)
    all_se = np.full((n_targets, n_bins, n_params), np.nan, dtype=np.float64)
    all_rmse = np.full((n_targets, n_bins), np.nan, dtype=np.float64)
    all_r2 = np.full((n_targets, n_bins), np.nan, dtype=np.float64)
    all_n_fitted = np.zeros((n_targets, n_bins), dtype=np.int64)
    all_status = np.zeros((n_targets, n_bins), dtype=np.int64)
    all_mean = np.full((n_targets, n_bins), np.nan, dtype=np.float64)
    all_std = np.full((n_targets, n_bins), np.nan, dtype=np.float64)
    all_entries = np.zeros((n_targets, n_bins), dtype=np.int64)

    t_l2_start = time.time()
    for ti in range(n_targets):
        solve_all_windows(
            XtX_all[ti], XtY_all[ti], n_all[ti], sum_y_all[ti], sum_y2_all[ti],
            nbr_indices, nbr_weights, nbr_counts,
            n_params, min_stat, _is_weighted_kernel,
            all_beta[ti], all_se[ti], all_rmse[ti], all_r2[ti],
            all_n_fitted[ti], all_status[ti],
            all_mean[ti], all_std[ti], all_entries[ti])
    t_l2 = time.time() - t_l2_start

    if verbose:
        print(f"[V5] Loop 2 (solve):  {t_l2:.4f}s  [cum: {time.time()-t0:.3f}s]")

    # --- Pack output --- flat arrays, no Python dicts, no per-bin loops
    t_pack_start = time.time()
    pred_names = [_sanitize_suffix(p) for p in linear_columns]
    expected_nbr = int(neighbor_offsets.shape[0]) if neighbor_offsets.size else 1

    out = {
        'bin_coords': bin_coords,
        'n_neighbors_used': nbr_counts.astype(np.int64),
        'n_rows_aggregated': all_entries[0].copy() if n_targets > 0 else np.zeros(n_bins, dtype=np.int64),
        'effective_window_fraction': nbr_counts.astype(np.float64) / expected_nbr if expected_nbr > 0 else np.full(n_bins, np.nan),
    }

    for ti, tgt in enumerate(fit_columns):
        # Coefficients
        if fit_intercept:
            out[f'{tgt}_intercept'] = all_beta[ti, :, 0].copy()
            out[f'{tgt}_intercept_err'] = all_se[ti, :, 0].copy()
            for pi, pname in enumerate(pred_names):
                out[f'{tgt}_slope_{pname}'] = all_beta[ti, :, pi + 1].copy()
                out[f'{tgt}_slope_{pname}_err'] = all_se[ti, :, pi + 1].copy()
        else:
            out[f'{tgt}_intercept'] = np.zeros(n_bins, dtype=np.float64)
            out[f'{tgt}_intercept_err'] = np.zeros(n_bins, dtype=np.float64)
            for pi, pname in enumerate(pred_names):
                out[f'{tgt}_slope_{pname}'] = all_beta[ti, :, pi].copy()
                out[f'{tgt}_slope_{pname}_err'] = all_se[ti, :, pi].copy()

        out[f'{tgt}_r_squared'] = all_r2[ti].copy()
        out[f'{tgt}_rmse'] = all_rmse[ti].copy()
        out[f'{tgt}_n_fitted'] = all_n_fitted[ti].copy()
        out[f'{tgt}_mean'] = all_mean[ti].copy()
        out[f'{tgt}_std'] = all_std[ti].copy()
        out[f'{tgt}_entries'] = all_entries[ti].copy()
    t_pack = time.time() - t_pack_start

    if verbose:
        print(f"[V5] Total: {time.time()-t0:.4f}s")

    if _collect_timings:
        out['_timings'] = {
            'setup': t_setup,
            'nbr_table': t_nbr,
            'loop1': t_l1,
            'loop2': t_l2,
            'pack': t_pack,
            'total': time.time() - t0,
        }

    return out


def _assemble_results_v5(
        v5_arrays: Dict[str, np.ndarray],
        gb_columns: List[str],
        fit_columns: List[str],
        linear_columns: List[str],
        suffix: str = '',
) -> pd.DataFrame:
    """Vectorized DataFrame assembly from V5 flat arrays. No per-bin loops."""
    bin_coords = v5_arrays['bin_coords']
    n_bins = bin_coords.shape[0]

    if n_bins == 0:
        return pd.DataFrame()

    # Start with bin coordinates — no suffix
    data = {}
    for d, dim in enumerate(gb_columns):
        data[dim] = bin_coords[:, d]

    pred_names = [_sanitize_suffix(p) for p in linear_columns]

    # Per-target columns
    for tgt in fit_columns:
        s = suffix
        data[f'{tgt}_mean{s}'] = v5_arrays[f'{tgt}_mean']
        data[f'{tgt}_std{s}'] = v5_arrays[f'{tgt}_std']
        data[f'{tgt}_median{s}'] = np.full(n_bins, np.nan)  # Cannot compute from sufficient stats
        data[f'{tgt}_entries{s}'] = v5_arrays[f'{tgt}_entries']
        data[f'{tgt}_intercept{s}'] = v5_arrays[f'{tgt}_intercept']
        data[f'{tgt}_intercept_err{s}'] = v5_arrays[f'{tgt}_intercept_err']
        for pname in pred_names:
            data[f'{tgt}_slope_{pname}{s}'] = v5_arrays[f'{tgt}_slope_{pname}']
            data[f'{tgt}_slope_{pname}_err{s}'] = v5_arrays[f'{tgt}_slope_{pname}_err']
        data[f'{tgt}_r_squared{s}'] = v5_arrays[f'{tgt}_r_squared']
        data[f'{tgt}_rmse{s}'] = v5_arrays[f'{tgt}_rmse']
        data[f'{tgt}_n_fitted{s}'] = v5_arrays[f'{tgt}_n_fitted']

    # Diagnostics
    data[f'quality_flag{s}'] = np.where(
        v5_arrays['n_rows_aggregated'] == 0, 'empty_window', '')
    data[f'n_neighbors_used{s}'] = v5_arrays['n_neighbors_used']
    data[f'n_rows_aggregated{s}'] = v5_arrays['n_rows_aggregated']
    data[f'effective_window_fraction{s}'] = v5_arrays['effective_window_fraction']

    return pd.DataFrame(data)


# ===============
# Assembly (V4 legacy — used by V1/V2/V3 paths)
# ===============

def _assemble_results(
        gb_columns: List[str],
        agg_results: List[_AggResult],
        fit_results: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]],
        fit_columns: List[str],
        linear_columns: List[str],
        agg_columns: Optional[List[str]] = None,
        agg_median: bool = False,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    _agg_cols = agg_columns or []

    # Build column order
    pred_suffixes = {p: _sanitize_suffix(p) for p in linear_columns}

    for ar in agg_results:
        base: Dict[str, Any] = {dim: ar.center[i] for i, dim in enumerate(gb_columns)}
        base["n_neighbors_used"] = ar.n_neighbors_used
        base["n_rows_aggregated"] = ar.n_rows_aggregated
        base["effective_window_fraction"] = ar.effective_window_fraction

        # Agg columns stats (COG etc.) — placed before fit_columns stats
        if _agg_cols and ar.agg_stats:
            for c in _agg_cols:
                cst = ar.agg_stats.get(c, {})
                base[f"{c}_mean"] = cst.get("mean", np.nan)
                base[f"{c}_std"] = cst.get("std", np.nan)
                if agg_median:
                    base[f"{c}_median"] = cst.get("median", np.nan)
        elif _agg_cols:
            for c in _agg_cols:
                base[f"{c}_mean"] = np.nan
                base[f"{c}_std"] = np.nan
                if agg_median:
                    base[f"{c}_median"] = np.nan

        # Aggregate stats (fit_columns)
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

    # Order columns: gb_columns -> agg_columns stats -> fit aggregations -> fit outputs -> diagnostics
    extra_agg_cols = []
    for c in _agg_cols:
        extra_agg_cols.append(f"{c}_mean")
        extra_agg_cols.append(f"{c}_std")
        if agg_median:
            extra_agg_cols.append(f"{c}_median")

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

    ordered = gb_columns + extra_agg_cols + agg_cols + fit_cols + diag_cols
    # Keep any other columns at the end (defensive)
    others = [c for c in out.columns if c not in ordered]
    out = out[ordered + others]

    return out


# =====================
# Metadata builder (V4-compatible + SW-specific fields)
# =====================

def _build_sw_metadata(
    fit_columns,
    linear_columns,
    suffix,
    fit_intercept,
    gb_columns,
    weights_column=None,
    min_stat=10,
    window_spec=None,
    boundary_mode=None,
    kernel='uniform',
    kernel_width=None,
    algorithm='recompute',
    backend_used='numpy',
    n_bins=0,
    computation_time_sec=0.0,
):
    """
    Build metadata dict for sliding window regression results.

    Flat structure following the V4 per-bin metadata convention.
    Same top-level keys as V4 (version, formulas, columns, parameters)
    plus SW-specific keys (window_spec, boundary_mode, kernel, algorithm,
    backend_used) at the same flat level.

    One level, one source of truth. V4 metadata is the reference standard;
    SW adds to it, does not restructure it.
    """
    metadata = {
        # --- V4-compatible keys (same convention) ---
        'version': '1.0',
        'formulas': {},
        'residual_formulas': {},
        'pull_formulas': {},
        'columns': {
            'gb_columns': list(gb_columns),
            'fit_columns': list(fit_columns),
            'linear_columns': list(linear_columns),
            'coefficients': {},
            'errors': {},
            'quality': {},
        },
        'parameters': {
            'suffix': suffix,
            'fit_intercept': fit_intercept,
            'min_stat': min_stat,
            'fit_type': 'sliding_window',
            'weights_column': weights_column,
        },
        # --- Flat keys from original SW metadata (SW-specific, not in sub-dicts) ---
        'window_spec': window_spec or {},
        'boundary_mode': boundary_mode or {},
        'kernel': kernel if isinstance(kernel, str) else 'custom',
        'kernel_width': kernel_width,
        'algorithm': algorithm,
        'backend_used': backend_used,
        'n_bins': n_bins,
        'computation_time_sec': computation_time_sec,
        'python_version': sys.version,
    }

    # agg_columns metadata (if present in kwargs-style caller)
    # Callers should pass agg_columns via metadata['agg_columns'] after build

    # Build per-target column lists and formulas
    for target in fit_columns:
        coef_cols = []
        err_cols = []

        if fit_intercept:
            coef_cols.append(f"{target}_intercept{suffix}")
            err_cols.append(f"{target}_intercept_err{suffix}")

        for col in linear_columns:
            coef_cols.append(f"{target}_slope_{col}{suffix}")
            err_cols.append(f"{target}_slope_{col}_err{suffix}")

        metadata['columns']['coefficients'][target] = coef_cols
        metadata['columns']['errors'][target] = err_cols
        metadata['columns']['quality'][target] = [
            f"{target}_rmse{suffix}",
            f"{target}_r_squared{suffix}",
            f"{target}_std{suffix}",
        ]

        # Prediction formula (same convention as V4)
        terms = []
        if fit_intercept:
            terms.append(f"{target}_intercept{suffix}")
        for col in linear_columns:
            terms.append(f"{target}_slope_{col}{suffix}*{col}")
        if terms:
            metadata['formulas'][f"{target}_pred{suffix}"] = " + ".join(terms)

    return metadata


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
        boundary: Union[str, Dict[str, str]] = 'full',
        kernel: Union[str, Callable] = 'uniform',
        kernel_width: Optional[Union[float, Dict[str, float]]] = None,
        aggregation_functions: Optional[Dict[str, List[str]]] = None,
        agg_columns: Optional[List[str]] = None,
        agg_median: bool = False,
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
    agg_columns : list[str], optional
        Additional columns to aggregate (mean, std, and optionally median)
        within each sliding window. Useful for computing center-of-gravity
        of groupby variables or predictor columns.

        Example: To get the data COG for the groupby coordinates::

            agg_columns = ['mpt', 'vertex_z', 'tgl', 'phi']

        Output: ``mpt_mean_sw``, ``mpt_std_sw``, ``vertex_z_mean_sw``, etc.

        When kernel is non-uniform (e.g. ``kernel='gaussian'``), mean and
        std are kernel-weighted. Median (if enabled) is always unweighted.
    agg_median : bool
        If True, also compute (unweighted) median for agg_columns.
        Default False — median computation requires sorting and can be
        slow for large windows.
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

    # Neighbor offsets (cheap — depends only on window_spec)
    neighbor_offsets = _generate_neighbor_offsets(full_window_spec, gb_columns)

    # Lightweight bounds (min/max per column) — needed for validation before dispatch.
    # Does NOT build bin_map (which is O(n_rows) Python and the V5tot bottleneck).
    _sel_df = df if selection is None else df[selection]
    bounds = {dim: (int(_sel_df[dim].min()), int(_sel_df[dim].max())) for dim in gb_columns}

    # Resolve V3b boundary and kernel (used by incremental path)
    boundary_resolved = _resolve_boundary(boundary, gb_columns)
    _validate_periodic_dims(boundary_resolved, bounds, full_window_spec)

    # Determine if non-uniform kernel is active
    _is_weighted_kernel = (kernel != 'uniform') if isinstance(kernel, str) else True
    kernel_width_resolved = _resolve_kernel_width(kernel_width, full_window_spec, gb_columns)
    kernel_width_vec = np.array([kernel_width_resolved[dim] for dim in gb_columns], dtype=np.float64)

    # Precompute offset weights (depends only on offsets, not on center)
    offset_weights = _precompute_offset_weights(neighbor_offsets, kernel, kernel_width_vec)

    # Fitting — dispatch by algorithm and backend
    if algorithm == 'incremental':
        # Dispatch: V5 (fast numpy-in/numpy-out) or V3 NumPy (fallback)
        _use_numba_v5 = (
            _resolved_backend == 'numba'
            and _check_numba_available()
            and weights is None  # WLS not supported in Numba kernel
        )

        if _use_numba_v5:
            # V5: fast path — DataFrame conversion once, then numpy only
            t_flat = time.time()
            bin_ids, X_all, Y_all, _n_bins, _bin_coords, _bounds = \
                _flatten_bins_for_v5(df, gb_columns, fit_columns, linear_columns, selection, fit_intercept)
            if verbose:
                print(f"[V5] Flatten: {_n_bins} bins, {X_all.shape[0]} rows, {time.time()-t_flat:.4f}s")

            v5_arrays = make_sliding_window_fit_v5_arrays(
                bin_ids=bin_ids, X_all=X_all, Y_all=Y_all,
                n_bins=_n_bins, bin_coords=_bin_coords, bounds=_bounds,
                gb_columns=gb_columns, fit_columns=fit_columns,
                linear_columns=linear_columns, window_spec=full_window_spec,
                boundary=boundary, kernel=kernel, kernel_width=kernel_width,
                fit_intercept=fit_intercept, min_stat=min_stat, verbose=verbose,
            )

            # Vectorized assembly — no per-bin Python loops
            t_asm = time.time()
            out = _assemble_results_v5(v5_arrays, gb_columns, fit_columns, linear_columns, suffix)

            # V5 agg_columns: compute from raw data using bin_ids mapping
            if agg_columns:
                _agg_cols_v5 = agg_columns
                _agg_arrays_v5 = {c: df[c].to_numpy(dtype=np.float64) for c in _agg_cols_v5}
                n_dims = len(gb_columns)

                # Build per-bin row lists from bin_ids — O(n_rows) once
                _bin_rows_v5: Dict[int, np.ndarray] = {}
                for bi in range(_n_bins):
                    _bin_rows_v5[bi] = np.where(bin_ids == bi)[0]

                # P1-2 fix: O(1) coord→bin_index lookup instead of O(n_bins) scan
                coord_to_bin: Dict[Tuple[int, ...], int] = {
                    tuple(int(_bin_coords[i, d]) for d in range(n_dims)): i
                    for i in range(_n_bins)
                }

                # Pre-allocate output arrays (avoid per-cell .loc assignment)
                _agg_out: Dict[str, np.ndarray] = {}
                for c in _agg_cols_v5:
                    _agg_out[f'{c}_mean'] = np.full(_n_bins, np.nan, dtype=np.float64)
                    _agg_out[f'{c}_std'] = np.full(_n_bins, np.nan, dtype=np.float64)
                    if agg_median:
                        _agg_out[f'{c}_median'] = np.full(_n_bins, np.nan, dtype=np.float64)

                # For each center bin, aggregate agg_columns over its window
                for bi in range(_n_bins):
                    center_coord = tuple(int(_bin_coords[bi, d]) for d in range(n_dims))
                    nbr_coords, valid_oi = _get_neighbor_bins_v2(
                        center_coord, neighbor_offsets, bounds, gb_columns,
                        boundary_resolved, full_window_spec)

                    # Collect rows and per-row kernel weights
                    idx_list_v5: List[int] = []
                    kw_list_v5: List[np.ndarray] = []  # kernel weight per row
                    for ni, nb_coord in enumerate(nbr_coords):
                        bj = coord_to_bin.get(nb_coord)
                        if bj is not None and bj in _bin_rows_v5:
                            rows_j = _bin_rows_v5[bj]
                            idx_list_v5.extend(rows_j.tolist())
                            # P1-3 fix: kernel weight for this neighbor
                            kw = float(offset_weights[valid_oi[ni]])
                            kw_list_v5.append(np.full(len(rows_j), kw, dtype=np.float64))

                    if not idx_list_v5:
                        continue

                    idx_v5 = np.array(idx_list_v5, dtype=np.int64)
                    kw_v5 = np.concatenate(kw_list_v5) if _is_weighted_kernel else None

                    for c in _agg_cols_v5:
                        y = _agg_arrays_v5[c][idx_v5]
                        y_fin = np.isfinite(y)
                        if _is_weighted_kernel and kw_v5 is not None:
                            valid = y_fin
                            x = y[valid]
                            ww = kw_v5[valid]
                            mean, std = _weighted_mean_std(x, ww)
                        else:
                            x = y[y_fin]
                            mean, std = _weighted_mean_std(x, None)

                        _agg_out[f'{c}_mean'][bi] = mean
                        _agg_out[f'{c}_std'][bi] = std
                        if agg_median:
                            _agg_out[f'{c}_median'][bi] = float(np.median(x)) if len(x) > 0 else np.nan

                # Assign columns to DataFrame at once (not per-cell)
                for key, arr in _agg_out.items():
                    out[f'{key}{suffix}'] = arr

            if verbose:
                print(f"[V5] Assembly: {time.time()-t_asm:.4f}s")

            _backend_used = "v5_numba"

            # Skip the old assemble path
            # Provenance (V4-compatible metadata + SW-specific fields)
            metadata = _build_sw_metadata(
                fit_columns=fit_columns,
                linear_columns=linear_columns,
                suffix=suffix,
                fit_intercept=fit_intercept,
                gb_columns=gb_columns,
                weights_column=weights,
                min_stat=min_stat,
                window_spec=full_window_spec,
                boundary_mode={dim: boundary_resolved[dim] for dim in gb_columns},
                kernel=kernel,
                kernel_width=kernel_width_resolved,
                algorithm=algorithm,
                backend_used=_backend_used,
                n_bins=_n_bins,
                computation_time_sec=time.time() - t0,
            )
            metadata['agg_columns'] = agg_columns or []
            metadata['agg_median'] = agg_median
            out.attrs.update(metadata)

            # Cast dtype
            if cast_dtype:
                float_cols = out.select_dtypes(include=[np.floating]).columns
                if len(float_cols) > 0:
                    out[float_cols] = out[float_cols].astype(cast_dtype)

            if verbose:
                print(f"[V5] Complete: {len(out)} bins, {time.time()-t0:.3f}s total")

            if return_metadata:
                return out, metadata
            return out

        else:
            # V3 NumPy fallback: needs bin_map (not used by V5 path)
            bin_map = _build_bin_index_map(df, gb_columns, selection)
            center_bins = list(bin_map.keys())

            # V3 pre-compute per-bin XtX/XtY, sum over window
            bin_suff_stats, agg_suff_stats = _precompute_bin_sufficient_stats(
                df=df,
                bin_map=bin_map,
                fit_columns=fit_columns,
                linear_columns=linear_columns,
                fit_intercept=fit_intercept,
                agg_columns=agg_columns,
            )

            # Lightweight aggregation
            agg_results = _compute_lightweight_agg_results(
                bin_map=bin_map,
                center_bins=center_bins,
                neighbor_offsets=neighbor_offsets,
                bounds=bounds,
                gb_columns=gb_columns,
                fit_columns=fit_columns,
                bin_suff_stats=bin_suff_stats,
                boundary_resolved=boundary_resolved,
                window_spec=full_window_spec,
                offset_weights=offset_weights if _is_weighted_kernel else None,
                use_weighted_kernel=_is_weighted_kernel,
                agg_columns=agg_columns,
                agg_suff_stats=agg_suff_stats,
                agg_median=agg_median,
            )

            if verbose:
                print(f"[SW] V3 numpy pre-compute done: {len(bin_map)} bins × "
                      f"{len(fit_columns)} targets, kernel={kernel}, "
                      f"boundary={boundary}, {time.time()-t0:.3f}s elapsed")

            fit_results = _fit_window_regression_incremental(
                bin_map=bin_map,
                bin_suff_stats=bin_suff_stats,
                center_bins=center_bins,
                neighbor_offsets=neighbor_offsets,
                bounds=bounds,
                fit_columns=fit_columns,
                linear_columns=linear_columns,
                fit_intercept=fit_intercept,
                min_stat=min_stat,
                agg_results=agg_results,
                boundary_resolved=boundary_resolved,
                window_spec=full_window_spec,
                offset_weights=offset_weights if _is_weighted_kernel else None,
                use_weighted_kernel=_is_weighted_kernel,
            )
            _backend_used = "incremental_numpy"

    else:
        # V1/V2 path: needs bin_map
        bin_map = _build_bin_index_map(df, gb_columns, selection)
        center_bins = list(bin_map.keys())

        # Row-level aggregation + recompute from raw data
        agg_results = _aggregate_window_zerocopy(
            df=df,
            bin_map=bin_map,
            center_bins=center_bins,
            neighbor_offsets=neighbor_offsets,
            bounds=bounds,
            gb_columns=gb_columns,
            fit_columns=fit_columns,
            weights=weights,
            agg_columns=agg_columns,
            agg_median=agg_median,
        )

        if verbose:
            print(f"[SW] Aggregation done: {len(agg_results)} bins, "
                  f"{time.time()-t0:.3f}s elapsed")

        # V1/V2 path: recompute from raw data
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
        agg_columns=agg_columns,
        agg_median=agg_median,
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

    # Provenance (V4-compatible metadata + SW-specific fields)
    metadata = _build_sw_metadata(
        fit_columns=fit_columns,
        linear_columns=linear_columns,
        suffix=suffix,
        fit_intercept=fit_intercept,
        gb_columns=gb_columns,
        weights_column=weights,
        min_stat=min_stat,
        window_spec=full_window_spec,
        boundary_mode={dim: boundary_resolved[dim] for dim in gb_columns},
        kernel=kernel,
        kernel_width=kernel_width_resolved,
        algorithm=algorithm,
        backend_used=_backend_used,
        n_bins=len(center_bins),
        computation_time_sec=time.time() - t0,
    )
    metadata['agg_columns'] = agg_columns or []
    metadata['agg_median'] = agg_median
    out.attrs.update(metadata)

    if verbose:
        print(f"[SW] Complete: {len(out)} bins, {time.time()-t0:.3f}s total")

    if return_metadata:
        return out, metadata
    return out


# ============================================================
# Strategy A: Split-column parallel sliding window regression
# ============================================================

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

_log = logging.getLogger(__name__)


def _counting_sort_indices(keys, n_groups):
    """O(N) counting sort: return (order, offsets).

    order[offsets[g]:offsets[g+1]] are the row indices belonging to group g.

    Uses Numba JIT if available, falls back to numpy otherwise.
    """
    try:
        return _counting_sort_indices_numba(keys, n_groups)
    except Exception:
        # Numba not available — fallback to numpy bincount + argsort
        n = len(keys)
        counts = np.bincount(keys.astype(np.int64), minlength=n_groups)
        offsets = np.zeros(n_groups + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        order = np.argsort(keys, kind='stable')
        return order, offsets


def _counting_sort_indices_numba(keys, n_groups):
    """Numba-accelerated O(N) counting sort."""
    import numba as nb

    @nb.njit(cache=True)
    def _csort(keys, n_groups):
        n = len(keys)
        # Count occurrences
        counts = np.zeros(n_groups, dtype=np.int64)
        for i in range(n):
            counts[keys[i]] += 1
        # Cumulative offsets
        offsets = np.zeros(n_groups + 1, dtype=np.int64)
        for g in range(n_groups):
            offsets[g + 1] = offsets[g] + counts[g]
        # Scatter into output order
        order = np.empty(n, dtype=np.int64)
        pos = offsets[:-1].copy()
        for i in range(n):
            g = keys[i]
            order[pos[g]] = i
            pos[g] += 1
        return order, offsets

    return _csort(keys.astype(np.int64), n_groups)


def _worker_init():
    """Set NUMBA_NUM_THREADS=1 inside worker to prevent oversubscription."""
    os.environ['NUMBA_NUM_THREADS'] = '1'
    try:
        import numba
        numba.config.THREADING_LAYER_PRIORITY = ['workqueue']
    except Exception:
        pass


# Module-level shared state for fork()-based parallel workers.
# Set by parent before spawning workers; children inherit via COW.
_shared_gb_arrays = None   # ndarray[N, n_gb], int64
_shared_x_array = None     # ndarray[N, n_pred], float64
_shared_y_array = None     # ndarray[N, n_tgt], float64
_shared_order = None       # ndarray[N], int64 — counting-sort order


def _worker_v5_shared(
        start,         # int — offset into _shared_order
        end,           # int — offset into _shared_order
        gb_columns,
        fit_columns,
        linear_columns,
        window_spec,
        fit_intercept,
        min_stat,
        boundary,
        kernel,
        kernel_width,
        suffix,
        unit_id,
        split_columns,
):
    """Worker: reads shared parent arrays via module globals, runs V5 path.

    With fork() start method, _shared_* globals are copy-on-write
    from the parent — zero pickle overhead for big arrays.
    Only (start, end) integers + small config args are pickled.
    """
    _worker_init()
    try:
        row_indices = _shared_order[start:end]
        gb_unit = _shared_gb_arrays[row_indices]
        x_unit = _shared_x_array[row_indices]
        y_unit = _shared_y_array[row_indices]

        # Reconstruct minimal DataFrame for _flatten_bins_for_v5
        data = {}
        for d, col in enumerate(gb_columns):
            data[col] = gb_unit[:, d]
        for i, col in enumerate(linear_columns):
            data[col] = x_unit[:, i]
        for i, col in enumerate(fit_columns):
            data[col] = y_unit[:, i]
        df_unit = pd.DataFrame(data)

        # Flatten to numpy
        bin_ids, X_all, Y_all, n_bins, bin_coords, bounds = \
            _flatten_bins_for_v5(df_unit, gb_columns, fit_columns,
                                 linear_columns, None, fit_intercept)

        if n_bins == 0:
            return (unit_id, pd.DataFrame())

        # V5 arrays path
        v5_out = make_sliding_window_fit_v5_arrays(
            bin_ids=bin_ids, X_all=X_all, Y_all=Y_all,
            n_bins=n_bins, bin_coords=bin_coords, bounds=bounds,
            gb_columns=gb_columns, fit_columns=fit_columns,
            linear_columns=linear_columns, window_spec=window_spec,
            boundary=boundary, kernel=kernel, kernel_width=kernel_width,
            fit_intercept=fit_intercept, min_stat=min_stat,
        )

        # Assemble DataFrame
        result = _assemble_results_v5(v5_out, gb_columns, fit_columns,
                                       linear_columns, suffix)

        # Add split_columns
        for col, val in zip(split_columns, unit_id):
            result[col] = val

        return (unit_id, result)

    except Exception as e:
        return (unit_id, str(e))


def _worker_v5(
        gb_vals,       # dict[str, ndarray] — gb columns for this unit
        x_vals,        # ndarray[n_rows, n_pred]
        y_vals,        # ndarray[n_rows, n_tgt]
        gb_columns,
        fit_columns,
        linear_columns,
        window_spec,
        fit_intercept,
        min_stat,
        boundary,
        kernel,
        kernel_width,
        suffix,
        unit_id,       # tuple — split_columns values for this unit
        split_columns,
):
    """Worker function: runs V5 arrays path on pre-extracted numpy arrays.

    Reconstructs a minimal DataFrame for _flatten_bins_for_v5, then calls
    V5 arrays path directly. Returns (unit_id, result_df) or (unit_id, error_str).
    """
    _worker_init()
    try:
        # Reconstruct minimal DataFrame from arrays
        data = {}
        for col in gb_columns:
            data[col] = gb_vals[col]
        for i, col in enumerate(linear_columns):
            data[col] = x_vals[:, i]
        for i, col in enumerate(fit_columns):
            data[col] = y_vals[:, i]
        df_unit = pd.DataFrame(data)

        # Flatten to numpy
        bin_ids, X_all, Y_all, n_bins, bin_coords, bounds = \
            _flatten_bins_for_v5(df_unit, gb_columns, fit_columns,
                                 linear_columns, None, fit_intercept)

        if n_bins == 0:
            return (unit_id, pd.DataFrame())

        # V5 arrays path
        v5_out = make_sliding_window_fit_v5_arrays(
            bin_ids=bin_ids, X_all=X_all, Y_all=Y_all,
            n_bins=n_bins, bin_coords=bin_coords, bounds=bounds,
            gb_columns=gb_columns, fit_columns=fit_columns,
            linear_columns=linear_columns, window_spec=window_spec,
            boundary=boundary, kernel=kernel, kernel_width=kernel_width,
            fit_intercept=fit_intercept, min_stat=min_stat,
        )

        # Assemble DataFrame
        result = _assemble_results_v5(v5_out, gb_columns, fit_columns,
                                       linear_columns, suffix)

        # Add split_columns
        for col, val in zip(split_columns, unit_id):
            result[col] = val

        return (unit_id, result)

    except Exception as e:
        return (unit_id, str(e))


def make_sliding_window_fit_parallel(
        df: pd.DataFrame,
        gb_columns: List[str],
        fit_columns: List[str],
        linear_columns: List[str],
        split_columns: List[str],
        n_workers: int,
        window_spec: Optional[Dict[str, int]] = None,
        weights: Optional[str] = None,
        suffix: str = '_sw',
        selection=None,
        fit_intercept: bool = True,
        min_stat: int = 10,
        cast_dtype: str = 'float64',
        backend: str = 'auto',
        boundary: Union[str, Dict[str, str]] = 'full',
        kernel: Union[str, Callable] = 'uniform',
        kernel_width: Optional[Union[float, Dict[str, float]]] = None,
        on_error: str = 'nan',
        verbose: int = 0,
) -> pd.DataFrame:
    """Parallel sliding window regression over independent data units.

    Splits the input DataFrame by `split_columns` (e.g., ['sector', 'stack']),
    dispatches each unit to a worker process, and concatenates results.
    Each worker runs the V5 numpy arrays path independently.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing gb_columns, fit_columns, linear_columns,
        and split_columns.
    gb_columns : list[str]
        Columns defining the bin grid within each unit.
    fit_columns : list[str]
        Target columns for regression.
    linear_columns : list[str]
        Predictor columns.
    split_columns : list[str]
        Columns defining independent parallel units (e.g., ['sector', 'stack']).
        Rows with the same split_columns values form one unit.
    n_workers : int
        Number of parallel worker processes. Should not exceed core count.
    window_spec : dict[str, int], optional
        Half-width of sliding window per gb dimension.
    weights : str, optional
        Not supported in parallel V5 path. Reserved for future use.
    suffix : str
        Suffix for output column names.
    selection : pd.Series, optional
        Boolean mask to filter rows before fitting.
    fit_intercept : bool
        Include intercept in regression.
    min_stat : int
        Minimum rows in window for valid fit.
    cast_dtype : str
        Output float dtype.
    backend : str
        Ignored — parallel always uses V5 Numba path.
    boundary : str or dict
        Boundary mode for window edge handling.
    kernel : str
        Weighting kernel.
    kernel_width : float or dict, optional
        Kernel width parameter.
    on_error : str
        Error handling: 'nan' (default) = return NaN for failed units,
        'raise' = raise on first failure.
    verbose : int
        0 = silent, 1 = summary, 2 = per-unit progress.

    Returns
    -------
    pd.DataFrame
        Same schema as make_sliding_window_fit, with split_columns added.
    """
    t0 = time.time()

    if window_spec is None:
        window_spec = {}
    full_window_spec = {dim: window_spec.get(dim, 0) for dim in gb_columns}

    if weights is not None:
        raise ValueError("weights not supported in parallel V5 path")

    # Validate split_columns exist
    for col in split_columns:
        if col not in df.columns:
            raise ValueError(f"split_column '{col}' not in DataFrame")
    for col in gb_columns:
        if col not in df.columns:
            raise ValueError(f"gb_column '{col}' not in DataFrame")

    # Apply selection
    if selection is not None:
        df_work = df[selection].copy()
    else:
        df_work = df

    # ---- Step 0: Extract columns as numpy arrays ONCE from DataFrame ----
    t0_extract = time.time()

    split_arrays = [df_work[c].to_numpy(dtype=np.int64) for c in split_columns]
    n_split = len(split_columns)

    # Compute integer split IDs (single key for multi-column split)
    if n_split == 1:
        split_ids = split_arrays[0]
    else:
        mins = [int(a.min()) for a in split_arrays]
        maxs = [int(a.max()) for a in split_arrays]
        sizes = [mx - mn + 1 for mn, mx in zip(mins, maxs)]
        shifted = [split_arrays[d] - mins[d] for d in range(n_split)]
        strides = np.ones(n_split, dtype=np.int64)
        for d in range(n_split - 2, -1, -1):
            strides[d] = strides[d + 1] * sizes[d + 1]
        split_ids = np.zeros(len(df_work), dtype=np.int64)
        for d in range(n_split):
            split_ids += shifted[d] * strides[d]

    n_rows_work = len(df_work)
    n_gb = len(gb_columns)

    gb_arrays = np.column_stack(
        [df_work[c].to_numpy(dtype=np.int64) for c in gb_columns]
    )
    x_array = np.column_stack(
        [df_work[c].to_numpy(dtype=np.float64) for c in linear_columns]
    ) if linear_columns else np.empty((n_rows_work, 0), dtype=np.float64)
    y_array = np.column_stack(
        [df_work[c].to_numpy(dtype=np.float64) for c in fit_columns]
    )

    t1_extract = time.time()

    # ---- Step 1: Counting sort — O(N), Numba JIT ----
    t0_sort = time.time()

    # Shift to 0-based keys for counting sort
    sid_min = int(split_ids.min())
    sid_max = int(split_ids.max())
    n_groups = sid_max - sid_min + 1
    keys = (split_ids - sid_min).astype(np.int64)

    order, offsets = _counting_sort_indices(keys, n_groups)

    # Identify non-empty groups and decode unit_ids
    tasks = []
    for g in range(n_groups):
        if offsets[g + 1] > offsets[g]:
            start, end = int(offsets[g]), int(offsets[g + 1])
            # Decode unit_id from original split_arrays
            first_row = int(order[start])
            unit_id = tuple(int(split_arrays[d][first_row])
                            for d in range(n_split))
            tasks.append((unit_id, start, end))
    n_units = len(tasks)

    t1_sort = time.time()

    if verbose >= 1:
        _log.info(
            f"[parallel] {n_units} units, {n_rows_work} rows: "
            f"extract={t1_extract - t0_extract:.3f}s "
            f"sort={t1_sort - t0_sort:.3f}s, "
            f"dispatching to {n_workers} workers")

    # ---- Step 2: Dispatch to workers (NO reorder — workers index directly) ----
    t0_workers = time.time()
    results = []
    errors = []

    if n_workers <= 1:
        # Serial fallback — index directly into original arrays
        for idx, (unit_id, start, end) in enumerate(tasks):
            row_idx = order[start:end]
            gb_unit = {c: gb_arrays[row_idx, d] for d, c in enumerate(gb_columns)}
            x_unit = x_array[row_idx]
            y_unit = y_array[row_idx]
            res = _worker_v5(
                gb_unit, x_unit, y_unit,
                gb_columns, fit_columns, linear_columns,
                full_window_spec, fit_intercept, min_stat,
                boundary, kernel, kernel_width,
                suffix, unit_id, split_columns,
            )
            uid, payload = res
            if isinstance(payload, str):
                errors.append((uid, payload))
                if on_error == 'raise':
                    raise RuntimeError(f"Unit {uid} failed: {payload}")
                if verbose >= 2:
                    _log.warning(f"[parallel] Unit {uid} failed: {payload}")
            else:
                results.append(payload)
            if verbose >= 2:
                _log.info(f"[parallel] {idx + 1}/{n_units} done")
    else:
        # Parallel: set module globals, fork() gives COW access to children.
        # Only (start, end) integers + small config are pickled per task.
        global _shared_gb_arrays, _shared_x_array, _shared_y_array, _shared_order
        _shared_gb_arrays = gb_arrays
        _shared_x_array = x_array
        _shared_y_array = y_array
        _shared_order = order

        try:
            futures = {}
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for unit_id, start, end in tasks:
                    fut = executor.submit(
                        _worker_v5_shared,
                        start, end,
                        gb_columns, fit_columns, linear_columns,
                        full_window_spec, fit_intercept, min_stat,
                        boundary, kernel, kernel_width,
                        suffix, unit_id, split_columns,
                    )
                    futures[fut] = unit_id

                done_count = 0
                for fut in as_completed(futures):
                    done_count += 1
                    uid = futures[fut]
                    try:
                        _, payload = fut.result()
                        if isinstance(payload, str):
                            errors.append((uid, payload))
                            if on_error == 'raise':
                                raise RuntimeError(f"Unit {uid} failed: {payload}")
                            if verbose >= 2:
                                _log.warning(f"[parallel] Unit {uid} failed: {payload}")
                        else:
                            results.append(payload)
                    except Exception as e:
                        errors.append((uid, str(e)))
                        if on_error == 'raise':
                            raise
                        if verbose >= 2:
                            _log.warning(f"[parallel] Unit {uid} exception: {e}")

                    if verbose >= 2 and done_count % max(1, n_units // 10) == 0:
                        _log.info(f"[parallel] {done_count}/{n_units} units complete")
        finally:
            _shared_gb_arrays = None
            _shared_x_array = None
            _shared_y_array = None
            _shared_order = None

    t1_workers = time.time()

    # ---- Step 4: Concatenate results ----
    t0_concat = time.time()
    if results:
        out = pd.concat(results, ignore_index=True)
    else:
        out = pd.DataFrame()

    # Cast dtype
    if cast_dtype and len(out) > 0:
        float_cols = out.select_dtypes(include=[np.floating]).columns
        if len(float_cols) > 0:
            out[float_cols] = out[float_cols].astype(cast_dtype)

    t1_concat = time.time()
    if verbose >= 1:
        _log.info(
            f"[parallel] Done: {len(out)} bins, {len(errors)} errors, "
            f"extract={t1_extract - t0_extract:.3f}s "
            f"sort={t1_sort - t0_sort:.3f}s "
            f"workers={t1_workers - t0_workers:.3f}s "
            f"concat={t1_concat - t0_concat:.3f}s "
            f"total={t1_concat - t0:.3f}s")

    if errors and verbose >= 1:
        for uid, err in errors:
            _log.warning(f"[parallel] Failed unit {uid}: {err}")

    return out
