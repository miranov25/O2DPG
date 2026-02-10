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
) -> Dict[str, Dict[Tuple[int, ...], _BinSuffStats]]:
    """Pre-compute XtX, XtY, n, Σy, Σy² for each bin and target.

    Returns
    -------
    dict[target_name, dict[bin_key, _BinSuffStats]]
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

    return result


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

            eff_frac = (n_used / expected_neighbors) if expected_neighbors > 0 else np.nan
            results.append(_AggResult(
                center=center, n_neighbors_used=n_used,
                n_rows_aggregated=n_rows_total, effective_window_fraction=eff_frac,
                stats=stats, row_indices=np.array([], dtype=np.int64),
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

        results.append(_AggResult(
            center=center,
            n_neighbors_used=n_used,
            n_rows_aggregated=n_rows_total,
            effective_window_fraction=eff_frac,
            stats=stats,
            row_indices=np.array([], dtype=np.int64),
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
    """
    n_pred = len(linear_columns)
    n_params = n_pred + (1 if fit_intercept else 0)
    use_v2_neighbors = (boundary_resolved is not None and window_spec is not None)

    out: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]] = {}

    for center in center_bins:
        if use_v2_neighbors:
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


def _pack_suff_stats_for_numba(
        bin_suff_stats: Dict[str, Dict[Tuple[int, ...], _BinSuffStats]],
        center_bins: List[Tuple[int, ...]],
        fit_columns: List[str],
        n_params: int,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Pack per-bin sufficient stats into contiguous arrays for Numba.

    Returns dict[target] → (XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all)
    where each array is indexed by bin position in center_bins.
    """
    n_bins = len(center_bins)
    bin_idx = {b: i for i, b in enumerate(center_bins)}

    result = {}
    for t in fit_columns:
        XtX_all = np.zeros((n_bins, n_params, n_params), dtype=np.float64)
        XtY_all = np.zeros((n_bins, n_params), dtype=np.float64)
        n_all = np.zeros(n_bins, dtype=np.int64)
        sum_y_all = np.zeros(n_bins, dtype=np.float64)
        sum_y2_all = np.zeros(n_bins, dtype=np.float64)

        target_stats = bin_suff_stats[t]
        for bk, bs in target_stats.items():
            if bk in bin_idx and bs.n > 0:
                i = bin_idx[bk]
                XtX_all[i] = bs.XtX
                XtY_all[i] = bs.XtY
                n_all[i] = bs.n
                sum_y_all[i] = bs.sum_y
                sum_y2_all[i] = bs.sum_y2

        result[t] = (XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all)

    return result


def _build_neighbor_table(
        center_bins: List[Tuple[int, ...]],
        neighbor_offsets: np.ndarray,
        bounds: Dict[str, Tuple[int, int]],
        boundary_resolved: Dict[str, str],
        window_spec: Dict[str, int],
        offset_weights: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build packed neighbor index + weight tables for Numba kernel.

    Returns:
        nbr_indices: (n_bins, max_neighbors) — index into center_bins, -1 = invalid
        nbr_weights: (n_bins, max_neighbors) — kernel weight, 0.0 for invalid
        nbr_counts:  (n_bins,) — number of valid neighbors per center
    """
    bin_idx = {b: i for i, b in enumerate(center_bins)}
    n_bins = len(center_bins)
    max_nbr = neighbor_offsets.shape[0]

    nbr_indices = np.full((n_bins, max_nbr), -1, dtype=np.int64)
    nbr_weights = np.zeros((n_bins, max_nbr), dtype=np.float64)
    nbr_counts = np.zeros(n_bins, dtype=np.int64)

    for i, center in enumerate(center_bins):
        neighbors, valid_idx = _get_neighbor_bins_v2(
            center, neighbor_offsets, bounds, boundary_resolved, window_spec)

        k = 0
        for j, nb in enumerate(neighbors):
            if nb in bin_idx:
                nbr_indices[i, k] = bin_idx[nb]
                nbr_weights[i, k] = offset_weights[valid_idx[j]]
                k += 1
        nbr_counts[i] = k

    return nbr_indices, nbr_weights, nbr_counts


def _get_numba_incremental_kernel():
    """JIT-compile the incremental solve kernel. Returns the @njit function."""
    import numba as nb

    @nb.njit(cache=True)
    def _solve_cholesky_inplace(A, b, p):
        """Solve A @ x = b via Cholesky decomposition (A must be SPD).

        Modifies A in place (lower triangle becomes L).
        Returns x in b, and success flag.
        """
        # Cholesky: A = L L^T
        for i in range(p):
            for j in range(i):
                s = 0.0
                for k in range(j):
                    s += A[i, k] * A[j, k]
                A[i, j] = (A[i, j] - s) / A[j, j]
            s = 0.0
            for k in range(i):
                s += A[i, k] * A[i, k]
            val = A[i, i] - s
            if val <= 0.0:
                return False  # Not positive definite
            A[i, i] = math.sqrt(val)

        # Forward: L @ z = b
        for i in range(p):
            s = 0.0
            for k in range(i):
                s += A[i, k] * b[k]
            b[i] = (b[i] - s) / A[i, i]

        # Backward: L^T @ x = z
        for i in range(p - 1, -1, -1):
            s = 0.0
            for k in range(i + 1, p):
                s += A[k, i] * b[k]
            b[i] = (b[i] - s) / A[i, i]

        return True

    @nb.njit(cache=True)
    def _cholesky_diag_inv(L, p):
        """Compute diag(A^-1) from Cholesky factor L where A = L L^T.

        Uses forward/backward substitution with unit vectors.
        Returns diagonal of A^-1.
        """
        diag = np.empty(p, dtype=np.float64)
        e = np.zeros(p, dtype=np.float64)

        for col in range(p):
            # Set up unit vector
            for i in range(p):
                e[i] = 0.0
            e[col] = 1.0

            # Forward: L z = e
            for i in range(p):
                s = 0.0
                for k in range(i):
                    s += L[i, k] * e[k]
                e[i] = (e[i] - s) / L[i, i]

            # Backward: L^T x = z
            for i in range(p - 1, -1, -1):
                s = 0.0
                for k in range(i + 1, p):
                    s += L[k, i] * e[k]
                e[i] = (e[i] - s) / L[i, i]

            diag[col] = e[col]

        return diag

    @nb.njit(cache=True, parallel=False)
    def incremental_solve_kernel(
            XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all,
            nbr_indices, nbr_weights, nbr_counts,
            n_params, min_stat, use_weighted_kernel,
            # outputs (pre-allocated):
            beta_out, se_out, rmse_out, r2_out, n_fitted_out, status_out,
    ):
        """Numba kernel: solve weighted normal equations for all center bins.

        For each center bin i:
          XtX_w = Σ_j w_j * XtX_all[nbr[j]]
          XtY_w = Σ_j w_j * XtY_all[nbr[j]]
          beta = solve(XtX_w, XtY_w) via Cholesky

        Parameters
        ----------
        status_out: 0 = OK, 1 = insufficient_stats, 2 = singular
        """
        n_bins = XtX_all.shape[0]
        p = n_params

        for i in range(n_bins):
            nc = nbr_counts[i]

            # Sum weighted sufficient stats
            XtX_w = np.zeros((p, p), dtype=np.float64)
            XtY_w = np.zeros(p, dtype=np.float64)
            n_total = 0
            sum_y = 0.0
            sum_y2 = 0.0
            w_n = 0.0  # weighted n for R² calculation

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

            if n_total < min_stat:
                status_out[i] = 1
                for r in range(p):
                    beta_out[i, r] = np.nan
                    se_out[i, r] = np.nan
                rmse_out[i] = np.nan
                r2_out[i] = np.nan
                continue

            # Solve via Cholesky (in-place on copy)
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

            # beta is now in b
            for r in range(p):
                beta_out[i, r] = b[r]

            # RSS = sum_y2 - beta . XtY
            rss = sum_y2
            for r in range(p):
                rss -= b[r] * XtY_w[r]
            if rss < 0.0:
                rss = 0.0

            # RMSE
            rmse_out[i] = math.sqrt(rss / n_total) if n_total > 0 else np.nan

            # R²
            if w_n > 0.0:
                y_mean = sum_y / w_n
            else:
                y_mean = 0.0
            ss_tot = sum_y2 - w_n * y_mean * y_mean
            if ss_tot > 0.0:
                r2_out[i] = 1.0 - rss / ss_tot
            else:
                r2_out[i] = np.nan

            # Standard errors (P1-3: NaN when weighted kernel)
            if use_weighted_kernel:
                for r in range(p):
                    se_out[i, r] = np.nan
            else:
                dof = n_total - p
                if dof > 0:
                    s2 = rss / dof
                    # Get diag(XtX^-1) from Cholesky factor A (=L)
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

    return incremental_solve_kernel


def _fit_window_regression_incremental_numba(
        bin_suff_stats: Dict[str, Dict[Tuple[int, ...], _BinSuffStats]],
        center_bins: List[Tuple[int, ...]],
        neighbor_offsets: np.ndarray,
        bounds: Dict[str, Tuple[int, int]],
        boundary_resolved: Dict[str, str],
        window_spec: Dict[str, int],
        offset_weights: np.ndarray,
        fit_columns: List[str],
        linear_columns: List[str],
        fit_intercept: bool,
        min_stat: int,
        use_weighted_kernel: bool,
        prebuilt_neighbor_table: Optional[Tuple] = None,
) -> Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]]:
    """V3-Numba: Solve normal equations via JIT-compiled Cholesky kernel.

    Packs all sufficient stats into contiguous arrays, builds a neighbor
    index table, and dispatches to a single Numba kernel for all bins.
    """
    n_pred = len(linear_columns)
    n_params = n_pred + (1 if fit_intercept else 0)
    n_bins = len(center_bins)

    # Pack sufficient stats
    packed = _pack_suff_stats_for_numba(bin_suff_stats, center_bins, fit_columns, n_params)

    # Reuse prebuilt neighbor table or build new one
    if prebuilt_neighbor_table is not None:
        nbr_indices, nbr_weights, nbr_counts = prebuilt_neighbor_table
    else:
        nbr_indices, nbr_weights, nbr_counts = _build_neighbor_table(
            center_bins, neighbor_offsets, bounds, boundary_resolved, window_spec, offset_weights)

    # Get JIT kernel
    kernel_fn = _get_numba_incremental_kernel()

    out: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]] = {}

    for t in fit_columns:
        XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all = packed[t]

        # Allocate outputs
        beta_out = np.full((n_bins, n_params), np.nan, dtype=np.float64)
        se_out = np.full((n_bins, n_params), np.nan, dtype=np.float64)
        rmse_out = np.full(n_bins, np.nan, dtype=np.float64)
        r2_out = np.full(n_bins, np.nan, dtype=np.float64)
        n_fitted_out = np.zeros(n_bins, dtype=np.int64)
        status_out = np.zeros(n_bins, dtype=np.int64)

        # Dispatch
        kernel_fn(
            XtX_all, XtY_all, n_all, sum_y_all, sum_y2_all,
            nbr_indices, nbr_weights, nbr_counts,
            n_params, min_stat, use_weighted_kernel,
            beta_out, se_out, rmse_out, r2_out, n_fitted_out, status_out,
        )

        # Unpack to dict structure
        for i, center in enumerate(center_bins):
            if center not in out:
                out[center] = {}

            if status_out[i] == 1:
                out[center][t] = _empty_fit_result("insufficient_stats", int(n_fitted_out[i]))
            elif status_out[i] == 2:
                out[center][t] = _empty_fit_result("singular_matrix", int(n_fitted_out[i]))
            else:
                if fit_intercept:
                    intercept = float(beta_out[i, 0])
                    intercept_err = float(se_out[i, 0])
                    coeffs = {linear_columns[j]: float(beta_out[i, j + 1]) for j in range(n_pred)}
                    coeffs_err = {linear_columns[j]: float(se_out[i, j + 1]) for j in range(n_pred)}
                else:
                    intercept = 0.0
                    intercept_err = 0.0
                    coeffs = {linear_columns[j]: float(beta_out[i, j]) for j in range(n_pred)}
                    coeffs_err = {linear_columns[j]: float(se_out[i, j]) for j in range(n_pred)}

                out[center][t] = {
                    "coeffs": coeffs,
                    "coeffs_err": coeffs_err,
                    "intercept": intercept,
                    "intercept_err": intercept_err,
                    "r_squared": float(r2_out[i]),
                    "rmse": float(rmse_out[i]),
                    "n_fitted": int(n_fitted_out[i]),
                    "quality_flag": "",
                }

    return out


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
        boundary: Union[str, Dict[str, str]] = 'full',
        kernel: Union[str, Callable] = 'uniform',
        kernel_width: Optional[Union[float, Dict[str, float]]] = None,
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
        # V3/V3b: pre-compute per-bin XtX/XtY, sum over window
        bin_suff_stats = _precompute_bin_sufficient_stats(
            df=df,
            bin_map=bin_map,
            fit_columns=fit_columns,
            linear_columns=linear_columns,
            fit_intercept=fit_intercept,
        )

        # Lightweight aggregation with V3b boundary + weights
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
        )

        if verbose:
            print(f"[SW] V3b pre-compute done: {len(bin_map)} bins × "
                  f"{len(fit_columns)} targets, kernel={kernel}, "
                  f"boundary={boundary}, {time.time()-t0:.3f}s elapsed")

        # Dispatch: Numba or NumPy for the solve loop
        _use_numba_incremental = (
            _resolved_backend == 'numba'
            and _check_numba_available()
            and weights is None  # WLS not supported in Numba kernel
        )

        if _use_numba_incremental:
            # Build neighbor table ONCE and share with agg + solve
            nbr_indices, nbr_weights_tbl, nbr_counts = _build_neighbor_table(
                center_bins, neighbor_offsets, bounds, boundary_resolved,
                full_window_spec, offset_weights)
            prebuilt_table = (nbr_indices, nbr_weights_tbl, nbr_counts)

            # Lightweight aggregation reusing prebuilt table
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
                prebuilt_neighbor_table=prebuilt_table,
            )

            if verbose:
                print(f"[SW] V3-Numba pre-compute done: {len(bin_map)} bins × "
                      f"{len(fit_columns)} targets, kernel={kernel}, "
                      f"boundary={boundary}, {time.time()-t0:.3f}s elapsed")

            fit_results = _fit_window_regression_incremental_numba(
                bin_suff_stats=bin_suff_stats,
                center_bins=center_bins,
                neighbor_offsets=neighbor_offsets,
                bounds=bounds,
                boundary_resolved=boundary_resolved,
                window_spec=full_window_spec,
                offset_weights=offset_weights,
                fit_columns=fit_columns,
                linear_columns=linear_columns,
                fit_intercept=fit_intercept,
                min_stat=min_stat,
                use_weighted_kernel=_is_weighted_kernel,
                prebuilt_neighbor_table=prebuilt_table,
            )
            _backend_used = "incremental_numba"
        else:
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
        # V1/V2 path: row-level aggregation + recompute from raw data
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
        "boundary_mode": {dim: boundary_resolved[dim] for dim in gb_columns},
        "kernel": kernel if isinstance(kernel, str) else "custom",
        "kernel_width": kernel_width_resolved,
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
