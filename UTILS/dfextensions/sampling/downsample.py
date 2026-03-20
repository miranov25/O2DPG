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

Sampling algorithm (Phase 13.11.DF v2.1 — architect's algorithm):
  Threshold-based efficiency sampling:
    Accept x_i if pdf(x_i) * U_i < threshold,  U_i ~ Uniform(0,1)
    Reconstruction weight: cw_i = 1 / max(pdf(x_i), threshold)
  Specify exactly one of frac or threshold (keyword-only after *).
  frac → threshold computed via bisection on empirical PDF.

PDF estimator (pdf_params):
  When pdf_params is provided, uses 3-layer estimator:
    Layer 1: Gaussian kernel smoothing (sigma = kernel_sigma_bins * dx)
    Layer 2: Poisson plug-in correction: * (1 - exp(-n_eff))
    Layer 3: Local polynomial regression at grid points
  Per-point evaluation via interpolation on corrected grid.
  Parameters accept scalar (all dims) or list (per continuous dim).
  Categorical dimensions are sliced, not fitted (AD-2/3/10).

Optional pdf_func:
  When provided, replaces empirical PDF entirely. Enables testing
  sampling/reweighting independently of PDF estimation.

Debug mode (debug=True):
  Adds columns: _debug_pdf, _debug_weight_raw, _debug_threshold.
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
    fit_coordinate: str = "x",
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
    fit_coordinate : str, default "x"
        Regression coordinate for Layer 3: "x" (data units) or "bin" (bin index).
        "bin" is recommended for non-uniform binning (e.g. quantile bins).

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
    if fit_coordinate == "bin":
        # For bin-index fitting: use median bin width for neighborhood size
        # This gives ~6 neighbors for quantile bins (vs 1 with edge bin width)
        dx_for_half = np.median(bin_widths)
    else:
        dx_for_half = dx
    half_width = max(1, int(round(poly_half_range / dx_for_half)))
    # Poisson-corrected counts for bin-index fitting (pdf_grid already has correction)
    counts_corrected = pdf_grid * N * bin_widths

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

        if fit_coordinate == "bin":
            # Fit Poisson-corrected counts vs bin index
            # For quantile bins: counts ≈ constant → polynomial fit trivial
            xc = nb[valid].astype(np.float64)
            x_eval = float(b)
            yc = counts_corrected[nb[valid]]
        else:
            xc = centers[nb[valid]]
            x_eval = centers[b]
            yc = pdf_grid[nb[valid]]

        actual_order = min(poly_order, len(xc) - 1)
        try:
            coeffs = np.polyfit(xc, yc, actual_order)
            val = np.polyval(coeffs, x_eval)
            if fit_coordinate == "bin":
                # Convert fitted counts back to pdf
                pdf_corrected[b] = max(val, 0.0) / (N * bin_widths[b])
            else:
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


# ===================================================================
# Threshold-based sampling (Phase 13.11.DF v2.1 — Architect's algorithm)
# ===================================================================

def _frac_from_threshold(pdf_at_points: np.ndarray, threshold: float) -> float:
    """
    Expected acceptance fraction for a given threshold.

    For each point: acceptance prob = min(1, threshold / pdf(x)).
    Returns mean acceptance probability = expected fraction.
    """
    pdf = np.maximum(pdf_at_points, 1e-30)
    acceptance = np.minimum(1.0, threshold / pdf)
    return float(acceptance.mean())


def _threshold_from_frac(pdf_at_points: np.ndarray, target_frac: float) -> float:
    """
    Find threshold that gives target acceptance fraction via bisection.

    Solves: mean(min(1, threshold / pdf(x))) = target_frac
    Monotonically increasing in threshold — bisection converges.
    ~20 iterations for machine precision.
    """
    if target_frac >= 1.0:
        return float(np.max(pdf_at_points))
    if target_frac <= 0.0:
        raise ValueError(f"target_frac must be in (0, 1], got {target_frac}")

    pdf = np.maximum(pdf_at_points, 1e-30)
    lo, hi = 0.0, float(np.max(pdf))

    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _frac_from_threshold(pdf, mid) < target_frac:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _threshold_sample(
    df: pd.DataFrame,
    pdf_at_points: np.ndarray,
    threshold: float,
    random_state: int,
    keep_weights: bool,
    weight_column: str,
    weight_dtype: np.dtype,
    debug: bool = False,
) -> pd.DataFrame:
    """
    Architect's threshold-based efficiency sampling.

    Accept point x_i if: pdf(x_i) * U_i < threshold,  U_i ~ Uniform(0,1)
    Reconstruction weight:  cw_i = 1 / max(pdf(x_i), threshold)

    Properties:
        - Points with pdf < threshold: always accepted, cw = 1/threshold
        - Points with pdf > threshold: accepted with prob threshold/pdf, cw = 1/pdf
        - Weight bound: 1/threshold (no extreme weights)
        - Unbiased: E[Σ cw_i] = N_orig

    Always adds to output:
        _pdf:       empirical PDF at each sampled point (per event)
        _threshold: threshold value used for this call (constant per group)

    With debug=True, additionally:
        _debug_pdf, _debug_weight_raw, _debug_threshold (legacy names)
    """
    rng = np.random.RandomState(random_state)
    N = len(df)
    pdf = pdf_at_points.astype(np.float64)

    # Accept/reject: accept if pdf(x) * U < threshold
    U = rng.uniform(0, 1, N)
    accepted = (pdf * U) < threshold
    chosen = np.where(accepted)[0]

    result = df.iloc[chosen].copy()

    # Reconstruction weight: 1 / max(pdf, threshold)
    pdf_capped = np.maximum(pdf[chosen], threshold)
    cw = 1.0 / pdf_capped

    if keep_weights:
        # Store normalized weight (consistent with downsampleDF convention)
        cw_norm = cw / cw.sum()
        result[weight_column] = cw_norm.astype(weight_dtype)

    # Production columns: always stored (needed for reconstruction)
    result["_pdf"] = pdf[chosen]
    result["_threshold"] = threshold

    if debug:
        result["_debug_pdf"] = pdf[chosen]
        result["_debug_weight_raw"] = cw  # 1/max(pdf, threshold)
        result["_debug_threshold"] = threshold

    # Always export threshold as metadata for downstream reconstruction
    result.attrs["threshold"] = threshold

    return result


def _get_pdf_at_points(
    work_df: pd.DataFrame,
    categorical_cols: list,
    continuous_specs: dict,
    pdf_params: Optional[dict],
    pdf_func: Optional[callable],
    method: str = "factorized",
) -> np.ndarray:
    """
    Get PDF values at each data point. Dispatches to pdf_func, factorized, or ND.
    """
    if pdf_func is not None:
        pdf_at_points = np.asarray(pdf_func(work_df), dtype=np.float64)
        return np.maximum(pdf_at_points, 1e-30)

    if method == "factorized":
        _, pdf_at_points, _ = _compute_smooth_weights_factorized(
            work_df, categorical_cols, continuous_specs,
            return_debug=True, pdf_params=pdf_params,
        )
    else:
        _, pdf_at_points, _ = _compute_smooth_weights_nd(
            work_df, categorical_cols, continuous_specs,
            return_debug=True, pdf_params=pdf_params,
        )
    return pdf_at_points


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
        fc = pdf_params.get("fit_coordinate", "x")

        for i, (col, edges) in enumerate(continuous_specs.items()):
            values = df[col].values.astype(np.float64)
            centers, pdf_1d = _estimate_pdf_smooth_1d(
                values, edges,
                kernel_sigma_bins=ks[i],
                bias_correction=bc,
                poly_order=int(po[i]),
                poly_half_range=pr[i],
                fit_coordinate=fc,
            )
            if fc == "bin":
                # Log-interpolation: interpolate log(pdf) in x-space, then exp()
                # For steeply falling PDFs (non-uniform bins), log(pdf) is nearly
                # linear between bin centers → log-interp is much more accurate
                log_pdf_1d = np.log(np.maximum(pdf_1d, 1e-30))
                pdf_at_points = np.exp(np.interp(values, centers, log_pdf_1d))
            else:
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
    variables: Dict[str, Union[Tuple, np.ndarray, str]],
    random_state: int,
    *,
    frac: Optional[float] = None,
    threshold: Optional[float] = None,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
    mask: Optional[Union[str, np.ndarray]] = None,
    debug: bool = False,
    pdf_params: Optional[dict] = None,
    pdf_func: Optional[callable] = None,
) -> pd.DataFrame:
    """
    Downsample with smooth factorized PDF weighting.

    PDF(x, y, cat, ...) ≈ PDF(cat) × PDF(x) × PDF(y) × ...

    Uses threshold-based efficiency sampling (architect's algorithm):
        Accept x_i if pdf(x_i) * U_i < threshold
        Reconstruction weight: cw_i = 1 / max(pdf(x_i), threshold)

    Specify exactly one of frac or threshold (keyword-only).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    variables : dict
        Per-variable spec:
        - 'categorical': exact groupby, no interpolation
        - (n_bins: int, lo, hi): Option C, uniform bins
        - list/np.ndarray of edges: Option D, explicit bins
    random_state : int
        Random seed for reproducibility.
    frac : float or None
        Target fraction of rows to retain. Threshold computed via bisection.
    threshold : float or None
        PDF threshold for accept/reject. Controls weight bound (1/threshold).
    keep_weights : bool, default True
    weight_dtype : np.dtype, default np.float32
    weight_column : str, default 'weight'
    mask : str, np.ndarray, or None
        Boolean selection. PDF estimated on masked rows only (AD-8).
    debug : bool, default False
        If True, add _debug_pdf, _debug_weight_raw, _debug_threshold columns.
    pdf_params : dict or None
        3-layer PDF estimator params (Phase 13.11.DF v2.1).
        If None, uses legacy estimator.
    pdf_func : callable or None
        If provided, replaces empirical PDF. Called as pdf_func(df) → array.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with reconstruction weights.

    Examples
    --------
    >>> variables = {'pT': (50, 0, 10), 'eta': [-2, -1, 0, 1, 2]}
    >>> out = downsampleDFSmoothFactorized(df, variables, 42, frac=0.1)
    >>> out = downsampleDFSmoothFactorized(df, variables, 42, threshold=0.01)
    """
    # Validate: exactly one of frac/threshold
    if (frac is None) == (threshold is None):
        raise ValueError("Specify exactly one of frac or threshold, not both/neither")
    if frac is not None and not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if threshold is not None and threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    if weight_column in df.columns:
        raise ValueError(f"weight_column '{weight_column}' already exists in DataFrame")

    categorical_cols, continuous_specs = _parse_variables(variables, df)

    work_df = _apply_mask(df, mask)
    work_df = _apply_range_filter(work_df, continuous_specs)

    if len(work_df) == 0:
        raise ValueError("No rows remain after mask and range filtering")

    # All-categorical edge case (AD-10): behave like downsampleDF
    if not continuous_specs:
        if not categorical_cols:
            raise ValueError("No variables specified")
        _frac = frac if frac is not None else 0.1
        return downsampleDF(
            work_df, frac=_frac, stratify=categorical_cols,
            random_state=random_state, keep_weights=keep_weights,
            weight_dtype=weight_dtype, weight_column=weight_column,
            debug=debug,
        )

    work_df = work_df.reset_index(drop=True)

    # Get PDF at each point
    pdf_at_points = _get_pdf_at_points(
        work_df, categorical_cols, continuous_specs,
        pdf_params, pdf_func, method="factorized",
    )

    # Resolve threshold from frac if needed
    if threshold is None:
        threshold = _threshold_from_frac(pdf_at_points, frac)

    return _threshold_sample(
        work_df, pdf_at_points, threshold, random_state,
        keep_weights, weight_column, weight_dtype, debug=debug,
    )


# ===================================================================
# downsampleDFSmooth — Smooth full ND PDF (v3.2 + debug)
# ===================================================================

def downsampleDFSmooth(
    df: pd.DataFrame,
    variables: Dict[str, Union[Tuple, np.ndarray, str]],
    random_state: int,
    *,
    frac: Optional[float] = None,
    threshold: Optional[float] = None,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
    mask: Optional[Union[str, np.ndarray]] = None,
    debug: bool = False,
    pdf_params: Optional[dict] = None,
    pdf_func: Optional[callable] = None,
) -> pd.DataFrame:
    """
    Downsample with smooth full ND PDF weighting.

    Per AD-3: one ND histogram. Categorical axes = exact lookup (groupby).
    Continuous axes = linear interpolation. Each categorical combination
    gets its own smooth PDF on the continuous variables.

    Uses threshold-based efficiency sampling (architect's algorithm).
    Specify exactly one of frac or threshold (keyword-only).

    Continuous dimension limit: D_continuous <= 5 per category combination.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    variables : dict
        Per-variable spec (same as downsampleDFSmoothFactorized).
    random_state : int
        Random seed for reproducibility.
    frac : float or None
        Target fraction. Threshold computed via bisection.
    threshold : float or None
        PDF threshold for accept/reject.
    keep_weights, weight_dtype, weight_column, mask, debug, pdf_params, pdf_func :
        Same as downsampleDFSmoothFactorized.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with reconstruction weights.

    Examples
    --------
    >>> variables = {'pT': (50, 0, 10), 'eta': (20, -2, 2)}
    >>> out = downsampleDFSmooth(df, variables, 42, frac=0.1)
    >>> out = downsampleDFSmooth(df, variables, 42, threshold=0.01)
    """
    if (frac is None) == (threshold is None):
        raise ValueError("Specify exactly one of frac or threshold, not both/neither")
    if frac is not None and not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if threshold is not None and threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
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
        _frac = frac if frac is not None else 0.1
        return downsampleDF(
            work_df, frac=_frac, stratify=categorical_cols,
            random_state=random_state, keep_weights=keep_weights,
            weight_dtype=weight_dtype, weight_column=weight_column,
            debug=debug,
        )

    work_df = work_df.reset_index(drop=True)

    pdf_at_points = _get_pdf_at_points(
        work_df, categorical_cols, continuous_specs,
        pdf_params, pdf_func, method="nd",
    )

    if threshold is None:
        threshold = _threshold_from_frac(pdf_at_points, frac)

    return _threshold_sample(
        work_df, pdf_at_points, threshold, random_state,
        keep_weights, weight_column, weight_dtype, debug=debug,
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

    Each trigger independently estimates its own PDF and samples using
    threshold-based efficiency sampling. Combined with OR bitmask.

    Each trigger dict must specify exactly one of 'frac' or 'threshold'.

    Maximum 16 triggers (uint16 bitmask).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    triggers : list of dict
        Each dict:
        - 'name': str — trigger label
        - 'variables': dict — per-variable spec
        - 'frac': float — target fraction (mutually exclusive with 'threshold')
        - 'threshold': float — PDF threshold (mutually exclusive with 'frac')
    random_state : int
        Base seed. Each trigger uses random_state + i.
    weight_dtype : np.dtype, default np.float32
    mask : str, np.ndarray, or None
        Boolean selection applied before all triggers (AD-8).
    pdf_params : dict or None
        3-layer PDF estimator params.

    Returns
    -------
    pd.DataFrame
        Rows selected by >= 1 trigger. Has weight_{name} columns and
        combined_trigger (uint16 bitmask).

    Examples
    --------
    >>> triggers = [
    ...     {'name': 'flat_pt', 'variables': {'pT': (50, 0, 10)}, 'frac': 0.07},
    ...     {'name': 'rare', 'variables': {'isDe': 'categorical'}, 'threshold': 0.01},
    ... ]
    >>> out = downsampleDFSmoothTrigger(df, triggers, random_state=42)
    """
    if len(triggers) > 16:
        raise ValueError(f"Maximum 16 triggers (uint16), got {len(triggers)}")

    for i, t in enumerate(triggers):
        if "name" not in t or "variables" not in t:
            raise ValueError(f"Trigger {i} missing 'name' or 'variables'")
        has_frac = "frac" in t
        has_thresh = "threshold" in t
        if has_frac == has_thresh:
            raise ValueError(
                f"Trigger '{t.get('name', i)}': specify exactly one of 'frac' or 'threshold'"
            )
        if has_frac and not (0 < t["frac"] <= 1):
            raise ValueError(f"Trigger '{t['name']}': frac must be in (0,1], got {t['frac']}")
        if has_thresh and t["threshold"] <= 0:
            raise ValueError(f"Trigger '{t['name']}': threshold must be > 0, got {t['threshold']}")

    # Apply mask once (AD-8)
    work_df = _apply_mask(df, mask)
    work_df = work_df.reset_index(drop=True)

    combined_trigger = np.zeros(len(work_df), dtype=int)
    trigger_bitmask = 1
    weight_columns = {}

    for i, trigger in enumerate(triggers):
        name = trigger["name"]
        variables = trigger["variables"]
        t_frac = trigger.get("frac", None)
        t_threshold = trigger.get("threshold", None)

        categorical_cols, continuous_specs = _parse_variables(variables, work_df)

        # Range filter per trigger
        trigger_df = _apply_range_filter(work_df, continuous_specs)
        if len(trigger_df) == 0:
            trigger_bitmask <<= 1
            continue

        trigger_df = trigger_df.reset_index(drop=False)  # keep original index
        orig_idx = trigger_df.index

        if not continuous_specs:
            # All-categorical (AD-10): groupby inverse-size, use old algorithm
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

            # Use old sampling for categorical
            _frac = t_frac if t_frac is not None else 0.1
            n_samples = int(len(trigger_df) * _frac)
            if n_samples == 0:
                trigger_bitmask <<= 1
                continue
            probs = weights / weights.sum()
            rng = np.random.RandomState(random_state + i)
            n_samples = min(n_samples, len(trigger_df))
            chosen_local = rng.choice(len(trigger_df), size=n_samples, replace=False, p=probs)
            chosen_global = trigger_df["index"].values[chosen_local]
            combined_trigger[chosen_global] |= trigger_bitmask

            w_col = np.full(len(work_df), np.nan, dtype=np.float64)
            w_col[trigger_df["index"].values] = probs
            weight_columns[f"weight_{name}"] = w_col.astype(weight_dtype)
        else:
            # Threshold-based sampling for continuous variables
            trigger_df_clean = trigger_df.drop(columns=["index"]).reset_index(drop=True)

            pdf_at_points = _get_pdf_at_points(
                trigger_df_clean, categorical_cols, continuous_specs,
                pdf_params, None, method="factorized",
            )

            # Resolve threshold
            if t_threshold is not None:
                thr = t_threshold
            else:
                thr = _threshold_from_frac(pdf_at_points, t_frac)

            # Accept/reject
            rng = np.random.RandomState(random_state + i)
            U = rng.uniform(0, 1, len(trigger_df_clean))
            accepted = (pdf_at_points * U) < thr
            chosen_local = np.where(accepted)[0]

            if len(chosen_local) == 0:
                trigger_bitmask <<= 1
                continue

            chosen_global = trigger_df["index"].values[chosen_local]
            combined_trigger[chosen_global] |= trigger_bitmask

            # Weights: 1 / max(pdf, threshold), normalized
            pdf_capped = np.maximum(pdf_at_points, thr)
            cw = 1.0 / pdf_capped
            cw_norm = cw / cw.sum()

            w_col = np.full(len(work_df), np.nan, dtype=np.float64)
            w_col[trigger_df["index"].values] = cw_norm
            weight_columns[f"weight_{name}"] = w_col.astype(weight_dtype)

        trigger_bitmask <<= 1

    # Build result
    result = work_df.copy()
    for col_name, col_data in weight_columns.items():
        result[col_name] = col_data
    result["combined_trigger"] = combined_trigger.astype(np.uint16)

    return result[result["combined_trigger"] > 0]
