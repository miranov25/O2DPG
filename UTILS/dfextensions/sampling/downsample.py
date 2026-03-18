"""
dfextensions/sampling/downsample.py

Stratified downsampling utilities for DataFrames.

Six public functions:
  Binned (v3.0 — ported from production):
    - downsampleDF:                   Binned groupby, inverse-group-size weights
    - downsampleDFTrigger:            Multi-trigger bitmask (binned)

  Smooth (v3.1/v3.2 + Phase 13.11.DF v2.1):
    - downsampleDFSmoothFactorized:   Smooth PDF, product of 1D marginals
    - downsampleDFSmooth:             Smooth PDF, full ND joint histogram
    - downsampleDFSmoothTrigger:      Multi-trigger bitmask (smooth)

Phase 13.10.DF v3.0 + Phase 13.11.DF v2.1 + Debug support

PDF estimator (pdf_params):
  When pdf_params is provided, uses 3-layer estimator:
    Layer 1: Gaussian kernel smoothing (sigma = kernel_sigma_bins * dx)
    Layer 2: Poisson plug-in correction: * (1 - exp(-n_eff))
    Layer 3: Local polynomial regression at grid points
  Per-point evaluation via interpolation on corrected grid.
  Parameters accept scalar (all dims) or list (per continuous dim).
  Categorical dimensions are sliced, not fitted (AD-2/3/10).

Debug mode (debug=True):
  Adds columns to output:
    - _debug_pdf:        Empirical PDF at each point (after interpolation)
    - _debug_weight_raw: 1/PDF before normalization (= raw inverse-PDF weight)
  
  These allow verification:
    - weight ∝ _debug_weight_raw (normalized)
    - _debug_pdf × _debug_weight_raw ≈ constant
"""

from typing import List, Union, Dict, Optional, Tuple
import numpy as np
import pandas as pd

__all__ = [
    "downsampleDF",
    "downsampleDFTrigger",
    "downsampleDFSmoothFactorized",
    "downsampleDFSmooth",
    "downsampleDFSmoothTrigger",
]


# ===================================================================
# Internal helpers — log-space interpolation
# ===================================================================

def _interpolate_empty_bins_log(pdf: np.ndarray) -> np.ndarray:
    """
    Fill empty (zero) bins via linear interpolation in log-space.

    Non-zero bins → log(pdf) → linearly interpolate at zero positions → exp().
    Positive-definite by construction. Empty bins get geometric mean of neighbors.
    """
    result = pdf.copy()
    nonzero = result > 0
    if nonzero.all() or not nonzero.any():
        return np.maximum(result, 1e-30)
    indices = np.arange(len(result))
    log_nonzero = np.log(result[nonzero])
    result = np.exp(np.interp(indices, indices[nonzero], log_nonzero))
    return result


def _interpolate_empty_bins_log_nd(pdf_nd: np.ndarray) -> np.ndarray:
    """
    Fill empty bins in an ND histogram via log-space interpolation.

    Applies 1D log-space interpolation along each axis sequentially.
    Approximate factorized fill — handles sparse edges/corners.
    """
    result = pdf_nd.copy()
    for axis in range(result.ndim):
        moved = np.moveaxis(result, axis, 0)
        shape = moved.shape
        flat = moved.reshape(shape[0], -1)
        for j in range(flat.shape[1]):
            col = flat[:, j]
            if (col == 0).any() and (col > 0).any():
                flat[:, j] = _interpolate_empty_bins_log(col)
        result = np.moveaxis(flat.reshape(shape), 0, axis)
    return np.maximum(result, 1e-30)


# ===================================================================
# Internal helpers — variable parsing (Phase 13.11.DF)
# ===================================================================

def _parse_variables(variables: dict, df: pd.DataFrame):
    """
    Parse variable specifications into categoricals and continuous.

    Variable spec types (AD-1, P0.1 disambiguation):
      - 'categorical': exact values, groupby, no interpolation
      - tuple (n_bins: int, lo, hi): Option C — uniform binning
      - list or np.ndarray: Option D — explicit bin edges

    Returns
    -------
    categorical_cols : list of str
    continuous_specs : dict of {col: bin_edges (np.ndarray)}
    """
    categorical_cols = []
    continuous_specs = {}

    for col, spec in variables.items():
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not in DataFrame")

        if isinstance(spec, str) and spec == 'categorical':
            categorical_cols.append(col)
        elif isinstance(spec, tuple) and len(spec) == 3 and isinstance(spec[0], int):
            # Option C: (n_bins, lo, hi) — tuple with int first element
            n_bins, lo, hi = spec
            if n_bins < 1:
                raise ValueError(f"Variable '{col}': n_bins must be >= 1, got {n_bins}")
            if lo >= hi:
                raise ValueError(f"Variable '{col}': lo must be < hi, got lo={lo}, hi={hi}")
            continuous_specs[col] = np.linspace(lo, hi, n_bins + 1)
        elif isinstance(spec, (list, np.ndarray)):
            # Option D: explicit bin edges — list or array, any length >= 2
            edges = np.asarray(spec, dtype=np.float64)
            if len(edges) < 2:
                raise ValueError(f"Variable '{col}': bin edges must have >= 2 elements")
            continuous_specs[col] = edges
        else:
            raise ValueError(
                f"Variable '{col}': spec must be 'categorical', "
                f"(n_bins: int, lo, hi), or list/array of bin edges. Got: {spec}"
            )

    return categorical_cols, continuous_specs


def _apply_mask(df: pd.DataFrame, mask) -> pd.DataFrame:
    """
    Apply mask to DataFrame. Returns view (no copy) of masked rows.

    Per AD-8: mask affects both PDF estimation and sampling.

    Parameters
    ----------
    mask : str, np.ndarray, or None
        - str: boolean column name in df
        - np.ndarray: boolean array of len(df)
        - None: no filtering
    """
    if mask is None:
        return df
    if isinstance(mask, str):
        if mask not in df.columns:
            raise ValueError(f"Mask column '{mask}' not in DataFrame")
        return df[df[mask].astype(bool)]
    if isinstance(mask, (np.ndarray, pd.Series)):
        if len(mask) != len(df):
            raise ValueError(f"Mask length ({len(mask)}) != DataFrame length ({len(df)})")
        return df[np.asarray(mask, dtype=bool)]
    raise ValueError(f"mask must be str, np.ndarray, or None. Got: {type(mask)}")


def _apply_range_filter(df: pd.DataFrame, continuous_specs: dict) -> pd.DataFrame:
    """
    Exclude rows where any continuous variable is outside its bin range.

    Per P0.2: out-of-range values excluded from both PDF estimation and sampling.
    Applied after mask.
    """
    if not continuous_specs:
        return df
    in_range = np.ones(len(df), dtype=bool)
    for col, edges in continuous_specs.items():
        vals = df[col].values
        in_range &= (vals >= edges[0]) & (vals <= edges[-1])
    return df[in_range]


def _estimate_1d_pdf_from_edges(values: np.ndarray, bin_edges: np.ndarray) -> tuple:
    """
    Estimate 1D PDF from data and given bin edges.

    Handles non-uniform bin widths. Empty bins filled via log-space interpolation.

    Returns bin_centers, pdf values.
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    pdf = counts / (len(values) * bin_widths)
    pdf = _interpolate_empty_bins_log(pdf)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, pdf


def _estimate_binned_pdf(
    values: np.ndarray,
    bin_edges: np.ndarray,
    bias_correction: bool = True,
) -> tuple:
    """
    Estimate 1D PDF via histogram (piecewise constant).

    Empty bins have PDF = 0 (AD-11). No interpolation.
    Optional first-order Poisson bias correction (Phase 13.11.DF v2.1):
        f_corr = f * (1 - exp(-n_bin))

    Uses plug-in estimator lambda_hat = n_bin (first-order Poisson debiasing).

    Returns bin_centers, pdf, counts.
    """
    counts, _ = np.histogram(values, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    N = len(values)
    pdf = counts.astype(np.float64) / (N * bin_widths)

    if bias_correction:
        # Plug-in correction: lambda_hat = n_bin (first-order Poisson debiasing)
        correction = 1.0 - np.exp(-counts.astype(np.float64))
        pdf = pdf * correction

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, pdf, counts


def _normalize_per_dim_param(param, n_dims, name):
    """
    Normalize a scalar or list to a per-dimension list.

    Parameters
    ----------
    param : scalar or list
        If scalar, replicated for all dimensions. If list, must have length n_dims.
    n_dims : int
    name : str
        For error messages.
    """
    if isinstance(param, (int, float)):
        return [param] * n_dims
    if len(param) != n_dims:
        raise ValueError(f"{name} has length {len(param)}, expected {n_dims}")
    return list(param)


def _estimate_pdf_smooth_1d(
    values: np.ndarray,
    bin_edges: np.ndarray,
    kernel_sigma_bins: float = 0.5,
    bias_correction: bool = True,
    poly_order: int = 2,
    poly_half_range: float = 0.5,
) -> tuple:
    """
    Three-layer 1D PDF estimator (Phase 13.11.DF v2.1).

    Layer 1: Gaussian kernel smoothing (sigma = kernel_sigma_bins * dx)
    Layer 2: Poisson plug-in correction: * (1 - exp(-n_eff))
    Layer 3: Local polynomial regression at each bin center

    Returns corrected PDF at bin centers. Per-point evaluation via
    np.interp (factorized) or RegularGridInterpolator (ND) in the caller.

    Parameters
    ----------
    values : np.ndarray
        Data values for one dimension.
    bin_edges : np.ndarray
        Histogram bin edges.
    kernel_sigma_bins : float, default 0.5
        Kernel sigma in bin-width units. 0.5 = half a bin.
    bias_correction : bool, default True
        Apply Poisson plug-in correction.
    poly_order : int, default 2
        Polynomial order: 1=linear, 2=parabolic.
    poly_half_range : float, default 0.5
        Fit neighborhood in data units (not bins).

    Returns
    -------
    bin_centers : np.ndarray
    pdf_corrected : np.ndarray
        Corrected PDF at bin centers.
    """
    from scipy.ndimage import gaussian_filter1d

    counts, _ = np.histogram(values, bins=bin_edges)
    bin_widths = np.diff(bin_edges)
    N = len(values)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dx = bin_widths[0]  # assume uniform for kernel
    n_bins = len(counts)

    # Layer 1: Gaussian kernel smoothing
    if kernel_sigma_bins > 0:
        counts_smooth = gaussian_filter1d(counts.astype(np.float64), sigma=kernel_sigma_bins)
        counts_smooth[counts_smooth < 0.0001 * counts_smooth.max()] = 0.0
    else:
        counts_smooth = counts.astype(np.float64)

    # Layer 2: Poisson correction
    pdf_grid = counts_smooth / (N * bin_widths)
    if bias_correction:
        correction = 1.0 - np.exp(-counts_smooth)
        pdf_grid = pdf_grid * correction

    # Layer 3: Local polynomial at each bin center
    half_width = max(1, int(round(poly_half_range / dx)))

    pdf_corrected = np.zeros(n_bins, dtype=np.float64)
    for b in range(n_bins):
        lo = max(0, b - half_width)
        hi = min(n_bins - 1, b + half_width)
        nb = np.arange(lo, hi + 1)

        valid = pdf_grid[nb] > 0
        if valid.sum() < (poly_order + 1):
            # Not enough points — fall back to grid value
            pdf_corrected[b] = max(pdf_grid[b], 0.0)
            continue

        xc = centers[nb[valid]]
        yc = pdf_grid[nb[valid]]

        actual_order = min(poly_order, len(xc) - 1)
        try:
            coeffs = np.polyfit(xc, yc, actual_order)
            val = np.polyval(coeffs, centers[b])
            pdf_corrected[b] = max(val, 0.0)
        except (np.linalg.LinAlgError, ValueError):
            pdf_corrected[b] = max(pdf_grid[b], 0.0)

    # Fill remaining zeros via log-interp (fallback for extreme tails)
    pdf_corrected = _interpolate_empty_bins_log(pdf_corrected)

    return centers, pdf_corrected


def _correct_nd_grid_polynomial(
    pdf_nd: np.ndarray,
    centers_per_axis: list,
    poly_order_per_axis: list,
    poly_half_range_per_axis: list,
    edges_per_axis: list,
) -> np.ndarray:
    """
    Apply local polynomial correction to an ND grid of PDF values.

    At each grid point, fits a local polynomial in the ND neighborhood
    and evaluates at the grid point itself, replacing the original value.

    For efficiency, correction is applied axis-by-axis (sequential 1D passes),
    not as a full joint ND polynomial. This is consistent with the factorized
    kernel smoothing approach and avoids combinatorial explosion in ND.

    Parameters
    ----------
    pdf_nd : np.ndarray
        ND array of PDF values at grid centers.
    centers_per_axis : list of np.ndarray
    poly_order_per_axis : list of int
    poly_half_range_per_axis : list of float (data units)
    edges_per_axis : list of np.ndarray

    Returns
    -------
    np.ndarray : corrected PDF grid
    """
    result = pdf_nd.copy()

    for axis in range(result.ndim):
        centers = centers_per_axis[axis]
        order = poly_order_per_axis[axis]
        dx = np.diff(edges_per_axis[axis])[0]
        half_width = max(1, int(round(poly_half_range_per_axis[axis] / dx)))
        n_bins = len(centers)

        # Move target axis to position 0 for easy iteration
        moved = np.moveaxis(result, axis, 0)
        shape = moved.shape
        flat = moved.reshape(shape[0], -1)

        for j in range(flat.shape[1]):
            col = flat[:, j].copy()
            corrected = np.zeros_like(col)

            for b in range(n_bins):
                lo = max(0, b - half_width)
                hi = min(n_bins - 1, b + half_width)
                nb = np.arange(lo, hi + 1)

                valid = col[nb] > 0
                if valid.sum() < (order + 1):
                    corrected[b] = max(col[b], 0.0)
                    continue

                xc = centers[nb[valid]]
                yc = col[nb[valid]]

                actual_order = min(order, len(xc) - 1)
                try:
                    coeffs = np.polyfit(xc, yc, actual_order)
                    val = np.polyval(coeffs, centers[b])
                    corrected[b] = max(val, 0.0)
                except (np.linalg.LinAlgError, ValueError):
                    corrected[b] = max(col[b], 0.0)

            flat[:, j] = corrected

        result = np.moveaxis(flat.reshape(shape), 0, axis)

    return result


def _weighted_sample(
    df: pd.DataFrame,
    weights: np.ndarray,
    n_samples: int,
    random_state: int,
    keep_weights: bool,
    weight_column: str,
    weight_dtype: np.dtype,
    debug: bool = False,
    debug_pdf: Optional[np.ndarray] = None,
    debug_weight_raw: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Sample rows using numpy (avoids pandas weight constraint).

    The stored weight column contains the **normalised sampling probability**
    p_i = w_i / sum(w), same convention as downsampleDF.
    To reconstruct the original distribution:
        correction_weight = 1 / p_i
        correction_weight *= N_orig / correction_weight.sum()
    
    When debug=True, also stores:
        _debug_pdf: empirical PDF at each sampled point
        _debug_weight_raw: 1/PDF before normalization
    """
    rng = np.random.RandomState(random_state)
    probs = weights / weights.sum()
    n_samples = min(n_samples, len(df))
    chosen = rng.choice(len(df), size=n_samples, replace=False, p=probs)
    result = df.iloc[chosen].copy()
    if keep_weights:
        result[weight_column] = probs[chosen].astype(weight_dtype)
    if debug:
        if debug_pdf is not None:
            result["_debug_pdf"] = debug_pdf[chosen].astype(np.float64)
        if debug_weight_raw is not None:
            result["_debug_weight_raw"] = debug_weight_raw[chosen].astype(np.float64)
    return result


def _compute_smooth_weights_factorized(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_specs: dict,
    return_debug: bool = False,
    pdf_params: Optional[dict] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute inverse-PDF weights using factorized (product of marginals) approach.

    PDF(x, y, cat, ...) = PDF(cat) × PDF_1d(x) × PDF_1d(y) × ...

    Each 1D marginal estimated independently.
    Categorical: value_counts normalized.
    
    When return_debug=True, returns (weights, pdf_at_points, weight_raw)
    
    Parameters
    ----------
    pdf_params : dict or None
        If provided, uses 3-layer estimator (Phase 13.11.DF v2.1):
          kernel_sigma_bins: float or list (default 0.5)
          bias_correction: bool (default True)
          poly_order: int or list (default 2)
          poly_half_range: float or list (default 0.5)
        If None, uses legacy estimator (_estimate_1d_pdf_from_edges).
    """
    log_pdf = np.zeros(len(df), dtype=np.float64)

    # Categorical marginals
    for col in categorical_cols:
        vc = df[col].value_counts(normalize=True)
        pdf_cat = df[col].map(vc).values.astype(np.float64)
        pdf_cat = np.maximum(pdf_cat, 1e-30)
        log_pdf += np.log(pdf_cat)

    # Continuous marginals
    cont_cols = list(continuous_specs.keys())
    n_cont = len(cont_cols)

    if pdf_params is not None and n_cont > 0:
        # 3-layer estimator per axis
        ks = _normalize_per_dim_param(pdf_params.get("kernel_sigma_bins", 0.5), n_cont, "kernel_sigma_bins")
        bc = pdf_params.get("bias_correction", True)
        po = _normalize_per_dim_param(pdf_params.get("poly_order", 2), n_cont, "poly_order")
        pr = _normalize_per_dim_param(pdf_params.get("poly_half_range", 0.5), n_cont, "poly_half_range")

        for i, (col, edges) in enumerate(continuous_specs.items()):
            values = df[col].values.astype(np.float64)
            centers, pdf_1d = _estimate_pdf_smooth_1d(
                values, edges,
                kernel_sigma_bins=ks[i],
                bias_correction=bc,
                poly_order=int(po[i]),
                poly_half_range=pr[i],
            )
            pdf_at_points = np.interp(values, centers, pdf_1d)
            pdf_at_points = np.maximum(pdf_at_points, 1e-30)
            log_pdf += np.log(pdf_at_points)
    else:
        # Legacy estimator
        for col, edges in continuous_specs.items():
            values = df[col].values.astype(np.float64)
            centers, pdf_1d = _estimate_1d_pdf_from_edges(values, edges)
            pdf_at_points = np.interp(values, centers, pdf_1d)
            pdf_at_points = np.maximum(pdf_at_points, 1e-30)
            log_pdf += np.log(pdf_at_points)

    # PDF at each point
    pdf = np.exp(log_pdf)
    
    # weight_raw = 1 / PDF (before any normalization)
    weight_raw = 1.0 / pdf
    
    # weights for sampling (shifted for numerical stability)
    log_weights = -log_pdf
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    
    if return_debug:
        return weights, pdf, weight_raw
    return weights


def _compute_smooth_weights_nd(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_specs: dict,
    return_debug: bool = False,
    pdf_params: Optional[dict] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute inverse-PDF weights using full ND histogram.

    Per AD-3: one ND histogram. Categorical axes = exact lookup (no interpolation).
    Continuous axes = linear interpolation. Implemented as: groupby categoricals,
    then ND histogram + interpolation on continuous axes per group.
    
    When return_debug=True, returns (weights, pdf_at_points, weight_raw)
    
    pdf_params: if provided, uses 3-layer estimator on the ND grid.
    """
    weights = np.zeros(len(df), dtype=np.float64)
    pdf_all = np.zeros(len(df), dtype=np.float64)
    weight_raw_all = np.zeros(len(df), dtype=np.float64)
    
    cont_cols = list(continuous_specs.keys())
    all_edges = [continuous_specs[c] for c in cont_cols]

    if not categorical_cols:
        # Pure continuous: single ND histogram
        w, pdf, wr = _compute_nd_weights_for_group(
            df, cont_cols, all_edges, return_debug=True, pdf_params=pdf_params)
        if return_debug:
            return w, pdf, wr
        return w
    else:
        # Per AD-3: group by categoricals, ND histogram per group
        grouped = df.groupby(categorical_cols)

        for cat_key, group_idx in grouped.groups.items():
            group_df = df.loc[group_idx]
            if len(group_df) == 0:
                continue

            if not cont_cols:
                # All-categorical per AD-10: uniform within group
                w = np.ones(len(group_df), dtype=np.float64)
                pdf = np.ones(len(group_df), dtype=np.float64) * (len(group_df) / len(df))
                wr = 1.0 / pdf
            else:
                w, pdf, wr = _compute_nd_weights_for_group(
                    group_df, cont_cols, all_edges, return_debug=True, pdf_params=pdf_params)

            # Scale by 1/group_fraction (categorical inverse weight)
            group_frac = len(group_df) / len(df)
            w /= group_frac
            pdf *= group_frac  # Adjust PDF for categorical contribution
            wr /= group_frac   # Adjust weight_raw accordingly

            idx = df.index.get_indexer(group_idx)
            weights[idx] = w
            pdf_all[idx] = pdf
            weight_raw_all[idx] = wr

    if return_debug:
        return weights, pdf_all, weight_raw_all
    return weights


def _compute_nd_weights_for_group(
    group_df: pd.DataFrame,
    cont_cols: list,
    all_edges: list,
    return_debug: bool = False,
    pdf_params: Optional[dict] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Compute inverse-PDF weights for one categorical group on continuous axes.
    
    When pdf_params is provided, applies 3-layer correction to the ND grid:
      1. Kernel smoothing per axis
      2. Poisson bias correction
      3. Local polynomial correction per axis
    Then uses RegularGridInterpolator on the corrected grid.
    """
    from scipy.interpolate import RegularGridInterpolator

    data_arrays = [group_df[c].values.astype(np.float64) for c in cont_cols]

    if len(cont_cols) == 0:
        w = np.ones(len(group_df), dtype=np.float64)
        if return_debug:
            return w, np.ones(len(group_df)), np.ones(len(group_df))
        return w

    hist_nd, _ = np.histogramdd(np.column_stack(data_arrays), bins=all_edges)

    # Non-uniform bin volumes
    widths_per_axis = [np.diff(e) for e in all_edges]
    if len(cont_cols) == 1:
        volumes = widths_per_axis[0]
    else:
        volumes = np.prod(
            np.meshgrid(*widths_per_axis, indexing='ij'), axis=0
        )

    N = len(group_df)
    n_cont = len(cont_cols)
    centers = [0.5 * (e[:-1] + e[1:]) for e in all_edges]

    if pdf_params is not None:
        from scipy.ndimage import gaussian_filter1d

        # Parse per-dimension parameters
        ks = _normalize_per_dim_param(pdf_params.get("kernel_sigma_bins", 0.5), n_cont, "kernel_sigma_bins")
        bc = pdf_params.get("bias_correction", True)
        po = _normalize_per_dim_param(pdf_params.get("poly_order", 2), n_cont, "poly_order")
        pr = _normalize_per_dim_param(pdf_params.get("poly_half_range", 0.5), n_cont, "poly_half_range")

        # Layer 1: Kernel smoothing per axis
        counts_smooth = hist_nd.astype(np.float64)
        for axis in range(n_cont):
            if ks[axis] > 0:
                counts_smooth = gaussian_filter1d(counts_smooth, sigma=ks[axis], axis=axis)
        # Floor tiny values
        max_val = counts_smooth.max()
        if max_val > 0:
            counts_smooth[counts_smooth < 0.0001 * max_val] = 0.0

        # Layer 2: Poisson correction
        pdf_nd = counts_smooth / (N * volumes)
        if bc:
            correction = 1.0 - np.exp(-counts_smooth)
            pdf_nd = pdf_nd * correction

        # Layer 3: Local polynomial correction per axis
        pdf_nd = _correct_nd_grid_polynomial(
            pdf_nd, centers,
            poly_order_per_axis=[int(p) for p in po],
            poly_half_range_per_axis=pr,
            edges_per_axis=all_edges,
        )

        # Fill remaining zeros
        pdf_nd = _interpolate_empty_bins_log_nd(pdf_nd)
    else:
        # Legacy path
        pdf_nd = hist_nd / (N * volumes)
        pdf_nd = _interpolate_empty_bins_log_nd(pdf_nd)

    interpolator = RegularGridInterpolator(
        centers, pdf_nd, method="linear",
        bounds_error=False, fill_value=None,
    )

    points = np.column_stack(data_arrays)
    if len(cont_cols) == 1:
        points = points.reshape(-1, 1)
    pdf_at_points = interpolator(points)
    pdf_at_points = np.maximum(pdf_at_points, 1e-30)

    # weight_raw = 1/PDF
    weight_raw = 1.0 / pdf_at_points
    
    log_weights = -np.log(pdf_at_points)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    
    if return_debug:
        return weights, pdf_at_points, weight_raw
    return weights


# ===================================================================
# downsampleDF — Binned groupby approach (v3.0, unchanged)
# ===================================================================

def downsampleDF(
    df: pd.DataFrame,
    frac: float,
    stratify: Union[str, List[str]],
    random_state: int,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
    debug: bool = False,
) -> pd.DataFrame:
    """
    Downsample a DataFrame with inverse-group-size weighting.

    Groups rows by ``stratify`` columns, assigns sampling weights inversely
    proportional to group size, normalizes to a probability distribution,
    and samples ``frac * len(df)`` rows without replacement.

    Ported from distortionMLFit.py (lines 116-147).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    frac : float
        Fraction of rows to retain (0 < frac <= 1).
    stratify : str or list of str
        Column(s) defining groups for stratified sampling.
    random_state : int
        Random seed for reproducibility. Required.
    keep_weights : bool, default True
        If True, include the sampling-weight column in the output.
    weight_dtype : np.dtype, default np.float32
        Data type for the weight column.
    weight_column : str, default 'weight'
        Name of the weight column added to the output.
    debug : bool, default False
        If True, add _debug_pdf and _debug_weight_raw columns.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with ``n = int(len(df) * frac)`` rows.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'g': ['A']*900 + ['B']*100, 'x': np.random.randn(1000)})
    >>> out = downsampleDF(df, frac=0.1, stratify='g', random_state=42)
    >>> len(out)
    100
    """
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if isinstance(stratify, str):
        stratify = [stratify]
    missing = [c for c in stratify if c not in df.columns]
    if missing:
        raise ValueError(f"stratify columns not in DataFrame: {missing}")
    if weight_column in df.columns:
        raise ValueError(f"weight_column '{weight_column}' already exists in DataFrame")

    group_sizes = df.groupby(stratify).size()
    
    # PDF per group = group_size / N (this is the empirical PDF for the binned approach)
    pdf_per_group = group_sizes / len(df)
    
    # Weight per group = 1 / group_size (unnormalized)
    weights_per_group = 1 / group_sizes
    weights_per_group_norm = weights_per_group / weights_per_group.sum()
    
    # Merge to get per-row values
    temp_df = df.merge(weights_per_group_norm.reset_index(name=weight_column), on=stratify, how="left")
    temp_df[weight_column] = temp_df[weight_column].astype(weight_dtype)
    
    if debug:
        # Add debug columns: PDF and weight_raw for each row
        temp_df = temp_df.merge(pdf_per_group.reset_index(name="_debug_pdf"), on=stratify, how="left")
        weight_raw_per_group = 1.0 / pdf_per_group
        temp_df = temp_df.merge(weight_raw_per_group.reset_index(name="_debug_weight_raw"), on=stratify, how="left")
    
    n_samples = int(len(df) * frac)
    downsampled_df = temp_df.sample(
        n=n_samples, weights=weight_column, replace=False, random_state=random_state,
    )
    if not keep_weights:
        downsampled_df = downsampled_df.drop(columns=[weight_column])
    if debug and not keep_weights:
        # Keep debug columns even if weights are dropped
        pass
    return downsampled_df


# ===================================================================
# downsampleDFTrigger — Multi-trigger bitmask (v3.0, unchanged)
# ===================================================================

def downsampleDFTrigger(
    df: pd.DataFrame,
    triggers: List[Dict],
    random_state: int,
    weight_dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    """
    Multi-trigger bitmask downsampling (binned approach).

    Ported from pidSkimmedFit.py (lines 224-261).
    Each trigger: groupby stratify → inverse-size weights → sample → bitmask OR.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    triggers : list of dict
        Each: {'name': str, 'stratify': str or list, 'frac': float}
    random_state : int
        Base seed. Each trigger uses random_state + i.
    weight_dtype : np.dtype, default np.float32

    Returns
    -------
    pd.DataFrame
        Rows selected by >= 1 trigger. Has weight_{name} and combined_trigger columns.

    Examples
    --------
    >>> triggers = [{'name': 't0', 'stratify': ['pt_bin'], 'frac': 0.1}]
    >>> out = downsampleDFTrigger(df, triggers, random_state=42)
    """
    if len(triggers) > 16:
        raise ValueError(f"Maximum 16 triggers (uint16), got {len(triggers)}")
    required_keys = {"name", "stratify", "frac"}
    for i, t in enumerate(triggers):
        missing_keys = required_keys - set(t.keys())
        if missing_keys:
            raise ValueError(f"Trigger {i} missing keys: {missing_keys}")
        if not (0 < t["frac"] <= 1):
            raise ValueError(f"Trigger '{t['name']}': frac must be in (0,1], got {t['frac']}")

    result = df.copy()
    combined_trigger = np.zeros(len(result), dtype=int)
    trigger_bitmask = 1
    for i, trigger in enumerate(triggers):
        name, stratify, frac = trigger["name"], trigger["stratify"], trigger["frac"]
        if isinstance(stratify, str):
            stratify = [stratify]
        missing = [c for c in stratify if c not in result.columns]
        if missing:
            raise ValueError(f"Trigger '{name}': columns not in DataFrame: {missing}")
        group_sizes = result.groupby(stratify).size()
        weights = (1 / group_sizes)
        weights /= weights.sum()
        weights = weights.astype(weight_dtype)
        weight_column = f"weight_{name}"
        result = result.merge(weights.reset_index(name=weight_column), on=stratify, how="left")
        n_samples = int(len(result) * frac)
        sampled_indices = result.sample(
            n=n_samples, weights=weight_column, replace=False, random_state=random_state + i,
        ).index
        combined_trigger[sampled_indices] |= trigger_bitmask
        trigger_bitmask <<= 1
    result["combined_trigger"] = combined_trigger.astype(np.uint16)
    return result[result["combined_trigger"] > 0]


# ===================================================================
# downsampleDFSmoothFactorized — Smooth factorized PDF (v3.2 + debug)
# ===================================================================

def downsampleDFSmoothFactorized(
    df: pd.DataFrame,
    frac: float,
    variables: Dict[str, Union[Tuple, np.ndarray, str]],
    random_state: int,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
    mask: Optional[Union[str, np.ndarray]] = None,
    debug: bool = False,
    pdf_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Downsample with smooth factorized PDF weighting.

    PDF(x, y, cat, ...) ≈ PDF(cat) × PDF(x) × PDF(y) × ...

    Each continuous marginal: histogram + linear interpolation.
    Categorical: value_counts. Product of marginals.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    frac : float
        Fraction of rows to retain (0 < frac <= 1).
    variables : dict
        Per-variable spec:
        - 'categorical': exact groupby, no interpolation
        - (n_bins: int, lo, hi): Option C, uniform bins
        - list/np.ndarray of edges: Option D, explicit bins
    random_state : int
        Random seed for reproducibility.
    keep_weights : bool, default True
    weight_dtype : np.dtype, default np.float32
    weight_column : str, default 'weight'
    mask : str, np.ndarray, or None
        Boolean selection. PDF estimated on masked rows only (AD-8).
    debug : bool, default False
        If True, add _debug_pdf and _debug_weight_raw columns to output.
    pdf_params : dict or None
        If provided, uses 3-layer PDF estimator (Phase 13.11.DF v2.1):
          kernel_sigma_bins: float or list (default 0.5)
          bias_correction: bool (default True)
          poly_order: int or list (default 2)
          poly_half_range: float or list (default 0.5)
        Per-dimension: pass list with one value per continuous dimension.
        If None, uses legacy estimator (log-interp + linear interp).

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with inverse-PDF weights.

    Examples
    --------
    >>> variables = {'type': 'categorical', 'pT': (50, 0, 10), 'eta': [-2, -1, 0, 1, 2]}
    >>> out = downsampleDFSmoothFactorized(df, frac=0.1, variables=variables, random_state=42)
    >>> # With 3-layer estimator:
    >>> out = downsampleDFSmoothFactorized(df, frac=0.1, variables=variables, random_state=42,
    ...     pdf_params={'poly_order': 2, 'poly_half_range': 0.5})
    """
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if weight_column in df.columns:
        raise ValueError(f"weight_column '{weight_column}' already exists in DataFrame")

    categorical_cols, continuous_specs = _parse_variables(variables, df)

    # Apply mask (AD-8: affects both PDF and sampling)
    work_df = _apply_mask(df, mask)
    # Apply range filter (P0.2: out-of-range excluded)
    work_df = _apply_range_filter(work_df, continuous_specs)

    if len(work_df) == 0:
        raise ValueError("No rows remain after mask and range filtering")

    # All-categorical edge case (AD-10): behave like downsampleDF
    if not continuous_specs:
        if not categorical_cols:
            raise ValueError("No variables specified")
        return downsampleDF(
            work_df, frac=frac, stratify=categorical_cols,
            random_state=random_state, keep_weights=keep_weights,
            weight_dtype=weight_dtype, weight_column=weight_column,
            debug=debug,
        )

    # Reset index for alignment
    work_df = work_df.reset_index(drop=True)
    
    if debug:
        weights, debug_pdf, debug_weight_raw = _compute_smooth_weights_factorized(
            work_df, categorical_cols, continuous_specs, return_debug=True, pdf_params=pdf_params
        )
    else:
        weights = _compute_smooth_weights_factorized(
            work_df, categorical_cols, continuous_specs, return_debug=False, pdf_params=pdf_params
        )
        debug_pdf = None
        debug_weight_raw = None
    
    n_samples = int(len(work_df) * frac)
    return _weighted_sample(
        work_df, weights, n_samples, random_state,
        keep_weights, weight_column, weight_dtype,
        debug=debug, debug_pdf=debug_pdf, debug_weight_raw=debug_weight_raw,
    )


# ===================================================================
# downsampleDFSmooth — Smooth full ND PDF (v3.2 + debug)
# ===================================================================

def downsampleDFSmooth(
    df: pd.DataFrame,
    frac: float,
    variables: Dict[str, Union[Tuple, np.ndarray, str]],
    random_state: int,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
    mask: Optional[Union[str, np.ndarray]] = None,
    debug: bool = False,
    pdf_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Downsample with smooth full ND PDF weighting.

    Per AD-3: one ND histogram. Categorical axes = exact lookup (groupby).
    Continuous axes = linear interpolation. Each categorical combination
    gets its own smooth PDF on the continuous variables.

    Continuous dimension limit: D_continuous <= 5 per category combination.
    Categorical dimensions partition the data — no limit.

    WARNING: Memory scales as n_category_combinations × prod(n_bins).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    frac : float
        Fraction of rows to retain (0 < frac <= 1).
    variables : dict
        Per-variable spec (same as downsampleDFSmoothFactorized).
    random_state : int
        Random seed for reproducibility.
    keep_weights : bool, default True
    weight_dtype : np.dtype, default np.float32
    weight_column : str, default 'weight'
    mask : str, np.ndarray, or None
        Boolean selection. PDF estimated on masked rows only (AD-8).
    debug : bool, default False
        If True, add _debug_pdf and _debug_weight_raw columns.
    pdf_params : dict or None
        3-layer PDF estimator params (same as downsampleDFSmoothFactorized).
        Applied to ND grid: kernel per axis, Poisson, local poly per axis.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with inverse-PDF weights.

    Examples
    --------
    >>> variables = {'type': 'categorical', 'pT': (50, 0, 10), 'eta': (20, -2, 2)}
    >>> out = downsampleDFSmooth(df, frac=0.1, variables=variables, random_state=42)
    """
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if weight_column in df.columns:
        raise ValueError(f"weight_column '{weight_column}' already exists in DataFrame")

    categorical_cols, continuous_specs = _parse_variables(variables, df)

    if len(continuous_specs) > 5:
        raise ValueError(
            f"downsampleDFSmooth supports <= 5 continuous dimensions, got {len(continuous_specs)}. "
            f"Use downsampleDFSmoothFactorized for higher dimensions."
        )

    work_df = _apply_mask(df, mask)
    work_df = _apply_range_filter(work_df, continuous_specs)

    if len(work_df) == 0:
        raise ValueError("No rows remain after mask and range filtering")

    # All-categorical (AD-10)
    if not continuous_specs:
        if not categorical_cols:
            raise ValueError("No variables specified")
        return downsampleDF(
            work_df, frac=frac, stratify=categorical_cols,
            random_state=random_state, keep_weights=keep_weights,
            weight_dtype=weight_dtype, weight_column=weight_column,
            debug=debug,
        )

    # Reset index for positional weight alignment
    work_df = work_df.reset_index(drop=True)
    
    if debug:
        weights, debug_pdf, debug_weight_raw = _compute_smooth_weights_nd(
            work_df, categorical_cols, continuous_specs, return_debug=True, pdf_params=pdf_params
        )
    else:
        weights = _compute_smooth_weights_nd(
            work_df, categorical_cols, continuous_specs, return_debug=False, pdf_params=pdf_params
        )
        debug_pdf = None
        debug_weight_raw = None
    
    n_samples = int(len(work_df) * frac)
    return _weighted_sample(
        work_df, weights, n_samples, random_state,
        keep_weights, weight_column, weight_dtype,
        debug=debug, debug_pdf=debug_pdf, debug_weight_raw=debug_weight_raw,
    )


# ===================================================================
# downsampleDFSmoothTrigger — Multi-trigger bitmask (smooth, v3.2)
# ===================================================================

def downsampleDFSmoothTrigger(
    df: pd.DataFrame,
    triggers: List[Dict],
    random_state: int,
    weight_dtype: np.dtype = np.float32,
    mask: Optional[Union[str, np.ndarray]] = None,
    pdf_params: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Multi-trigger bitmask downsampling with smooth PDF.

    Each trigger independently estimates its own PDF (smooth or categorical)
    and samples. Combined with OR bitmask.

    Weights are trigger-specific; no universal combined weight is defined.
    Downstream code must choose the relevant weight_{name} column.

    Maximum 16 triggers (uint16 bitmask).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    triggers : list of dict
        Each dict:
        - 'name': str — trigger label
        - 'variables': dict — per-variable spec (Option C/D/categorical)
        - 'frac': float — fraction to sample
    random_state : int
        Base seed. Each trigger uses random_state + i.
    weight_dtype : np.dtype, default np.float32
    mask : str, np.ndarray, or None
        Boolean selection applied before all triggers (AD-8).

    Returns
    -------
    pd.DataFrame
        Rows selected by >= 1 trigger. Has weight_{name} columns and
        combined_trigger (uint16 bitmask).

    Examples
    --------
    >>> triggers = [
    ...     {'name': 'flat_pid_pt', 'variables': {
    ...         'type': 'categorical', 'fPidIndex': 'categorical',
    ...         'fSigned1Pt': (50, -5, 5),
    ...     }, 'frac': 0.07},
    ...     {'name': 'rare_De', 'variables': {'isDe': 'categorical'}, 'frac': 0.01},
    ... ]
    >>> out = downsampleDFSmoothTrigger(df, triggers, random_state=42, mask='isForFit')
    """
    if len(triggers) > 16:
        raise ValueError(f"Maximum 16 triggers (uint16), got {len(triggers)}")

    required_keys = {"name", "variables", "frac"}
    for i, t in enumerate(triggers):
        missing_keys = required_keys - set(t.keys())
        if missing_keys:
            raise ValueError(f"Trigger {i} missing keys: {missing_keys}")
        if not (0 < t["frac"] <= 1):
            raise ValueError(f"Trigger '{t['name']}': frac must be in (0,1], got {t['frac']}")

    # Apply mask once (AD-8)
    work_df = _apply_mask(df, mask)
    work_df = work_df.reset_index(drop=True)

    combined_trigger = np.zeros(len(work_df), dtype=int)
    trigger_bitmask = 1
    weight_columns = {}

    for i, trigger in enumerate(triggers):
        name = trigger["name"]
        variables = trigger["variables"]
        frac = trigger["frac"]

        categorical_cols, continuous_specs = _parse_variables(variables, work_df)

        # Range filter per trigger (different triggers may have different ranges)
        trigger_df = _apply_range_filter(work_df, continuous_specs)
        if len(trigger_df) == 0:
            trigger_bitmask <<= 1
            continue

        # Compute weights
        if not continuous_specs:
            # All-categorical (AD-10): groupby inverse-size
            if categorical_cols:
                group_sizes = trigger_df.groupby(categorical_cols).size()
                w = 1 / group_sizes
                w /= w.sum()
                merged = trigger_df.merge(
                    w.reset_index(name=f"_w_{name}"), on=categorical_cols, how="left"
                )
                weights = merged[f"_w_{name}"].values.astype(np.float64)
            else:
                weights = np.ones(len(trigger_df), dtype=np.float64)
        else:
            weights = _compute_smooth_weights_factorized(
                trigger_df, categorical_cols, continuous_specs, pdf_params=pdf_params
            )

        # Sample
        n_samples = int(len(trigger_df) * frac)
        if n_samples == 0:
            trigger_bitmask <<= 1
            continue

        probs = weights / weights.sum()
        rng = np.random.RandomState(random_state + i)
        n_samples = min(n_samples, len(trigger_df))
        chosen_local = rng.choice(len(trigger_df), size=n_samples, replace=False, p=probs)

        # Map back to work_df indices
        chosen_global = trigger_df.index[chosen_local]
        combined_trigger[chosen_global] |= trigger_bitmask

        # Store weights for all rows (selected or not)
        w_col = np.full(len(work_df), np.nan, dtype=np.float64)
        w_col[trigger_df.index] = probs
        weight_columns[f"weight_{name}"] = w_col.astype(weight_dtype)

        trigger_bitmask <<= 1

    # Build result
    result = work_df.copy()
    for col_name, col_data in weight_columns.items():
        result[col_name] = col_data
    result["combined_trigger"] = combined_trigger.astype(np.uint16)

    return result[result["combined_trigger"] > 0]
