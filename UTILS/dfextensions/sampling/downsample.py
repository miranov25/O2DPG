"""
dfextensions/sampling/downsample.py

Stratified downsampling utilities for DataFrames.

Four approaches:
  - downsampleDF:                   Binned groupby, inverse-group-size weights
  - downsampleDFTrigger:            Multi-trigger bitmask (binned)
  - downsampleDFSmoothFactorized:   Smooth PDF, product of 1D marginals
  - downsampleDFSmooth:             Smooth PDF, full ND joint histogram

Ported from distortionMLFit.py and pidSkimmedFit.py.
Phase 13.10.DF v3.0+v3.1
"""

from typing import List, Union, Dict
import numpy as np
import pandas as pd

__all__ = [
    "downsampleDF",
    "downsampleDFTrigger",
    "downsampleDFSmoothFactorized",
    "downsampleDFSmooth",
]


# ===================================================================
# Internal helpers
# ===================================================================

def _interpolate_empty_bins_log(pdf: np.ndarray) -> np.ndarray:
    """
    Fill empty (zero) bins via linear interpolation in log-space.

    Non-zero bins: compute log(pdf).
    Zero bins:     linearly interpolate log(pdf) from neighbors, then exp().

    This guarantees positive-definiteness: exp(interpolated) > 0 always.
    Empty bins get approximately the geometric mean of their neighbors.

    Parameters
    ----------
    pdf : np.ndarray
        1D PDF array. May contain zeros.

    Returns
    -------
    np.ndarray
        PDF with zeros filled by log-space interpolation. Always > 0.
    """
    result = pdf.copy()
    nonzero = result > 0
    if nonzero.all() or not nonzero.any():
        return np.maximum(result, 1e-30)

    indices = np.arange(len(result))
    log_nonzero = np.log(result[nonzero])
    # Interpolate in log-space; extrapolate edges using nearest value
    result = np.exp(np.interp(indices, indices[nonzero], log_nonzero))
    return result


def _interpolate_empty_bins_log_nd(pdf_nd: np.ndarray) -> np.ndarray:
    """
    Fill empty bins in an ND histogram via log-space interpolation.

    Applies 1D log-space interpolation along each axis sequentially.
    This is an approximate factorized fill — not exact for correlated
    empty regions, but handles the common case of sparse edges/corners.

    Parameters
    ----------
    pdf_nd : np.ndarray
        ND PDF array. May contain zeros.

    Returns
    -------
    np.ndarray
        PDF with zeros filled. Always > 0.
    """
    result = pdf_nd.copy()
    # Apply 1D interpolation along each axis
    for axis in range(result.ndim):
        # Move target axis to position 0, flatten the rest
        moved = np.moveaxis(result, axis, 0)
        shape = moved.shape
        flat = moved.reshape(shape[0], -1)
        for j in range(flat.shape[1]):
            col = flat[:, j]
            if (col == 0).any() and (col > 0).any():
                flat[:, j] = _interpolate_empty_bins_log(col)
        result = np.moveaxis(flat.reshape(shape), 0, axis)
    # Safety floor for any remaining zeros (all-zero slices)
    return np.maximum(result, 1e-30)


def _estimate_1d_pdf(
    values: np.ndarray,
    n_bins: int,
    clip_quantile: float = 0.001,
) -> tuple:
    """
    Estimate a 1D PDF via histogram with log-space empty-bin interpolation.

    Returns bin_centers, pdf values, and bin_edges.
    Range is clipped by quantile to avoid sparse tails.
    Empty bins are filled via log-space interpolation (positive-definite).
    """
    lo = np.quantile(values, clip_quantile)
    hi = np.quantile(values, 1.0 - clip_quantile)
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    pdf = counts / (len(values) * bin_width)
    # Fill empty bins via log-space interpolation
    pdf = _interpolate_empty_bins_log(pdf)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, pdf, bin_edges


def _weighted_sample(
    df: pd.DataFrame,
    weights: np.ndarray,
    n_samples: int,
    random_state: int,
    keep_weights: bool,
    weight_column: str,
    weight_dtype: np.dtype,
) -> pd.DataFrame:
    """
    Sample rows using numpy (avoids pandas weight constraint).

    Uses np.random.choice with normalised weights, then builds
    the output DataFrame.

    The stored weight column contains the **normalised sampling probability**
    p_i = w_i / sum(w), same convention as downsampleDF.
    To reconstruct the original distribution:
        correction_weight = 1 / p_i
        correction_weight *= N_orig / correction_weight.sum()
    """
    rng = np.random.RandomState(random_state)
    probs = weights / weights.sum()
    chosen = rng.choice(len(df), size=n_samples, replace=False, p=probs)
    result = df.iloc[chosen].copy()
    if keep_weights:
        result[weight_column] = probs[chosen].astype(weight_dtype)
    return result


# ===================================================================
# downsampleDF — Binned groupby approach (v3.0)
# ===================================================================

def downsampleDF(
    df: pd.DataFrame,
    frac: float,
    stratify: Union[str, List[str]],
    random_state: int,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
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

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with ``n = int(len(df) * frac)`` rows.

    Raises
    ------
    ValueError
        If ``frac`` is not in (0, 1], if ``stratify`` columns are missing,
        or if ``weight_column`` already exists in *df*.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'g': ['A']*900 + ['B']*100, 'x': np.random.randn(1000)})
    >>> out = downsampleDF(df, frac=0.1, stratify='g', random_state=42)
    >>> len(out)
    100
    """
    # --- input validation ---------------------------------------------------
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    if isinstance(stratify, str):
        stratify = [stratify]

    missing = [c for c in stratify if c not in df.columns]
    if missing:
        raise ValueError(f"stratify columns not in DataFrame: {missing}")

    if weight_column in df.columns:
        raise ValueError(
            f"weight_column '{weight_column}' already exists in DataFrame"
        )

    # --- algorithm (unchanged from distortionMLFit.py) ----------------------
    group_sizes = df.groupby(stratify).size()
    weights = 1 / group_sizes
    weights /= weights.sum()

    temp_df = df.merge(
        weights.reset_index(name=weight_column), on=stratify, how="left"
    )
    temp_df[weight_column] = temp_df[weight_column].astype(weight_dtype)

    n_samples = int(len(df) * frac)
    downsampled_df = temp_df.sample(
        n=n_samples, weights=weight_column, replace=False,
        random_state=random_state,
    )

    if not keep_weights:
        downsampled_df = downsampled_df.drop(columns=[weight_column])

    return downsampled_df


# ===================================================================
# downsampleDFTrigger — Multi-trigger bitmask (v3.0)
# ===================================================================

def downsampleDFTrigger(
    df: pd.DataFrame,
    triggers: List[Dict],
    random_state: int,
    weight_dtype: np.dtype = np.float32,
) -> pd.DataFrame:
    """
    Multi-trigger bitmask downsampling.

    Each trigger independently samples rows according to its own stratification
    and fraction.  A combined bitmask records which trigger(s) selected each
    row.  Only rows selected by at least one trigger are returned.

    Ported from pidSkimmedFit.py (lines 224-261).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    triggers : list of dict
        Each dict must contain:
        - ``'name'``     : str — trigger label
        - ``'stratify'`` : str or list of str — grouping columns
        - ``'frac'``     : float — fraction to sample (0 < frac <= 1)
    random_state : int
        Base random seed.  Each trigger uses ``random_state + i``.
    weight_dtype : np.dtype, default np.float32
        Data type for per-trigger weight columns.

    Returns
    -------
    pd.DataFrame
        Rows selected by at least one trigger, with ``weight_{name}``
        columns and ``combined_trigger`` bitmask (uint16).

    Raises
    ------
    ValueError
        If trigger dicts are missing keys, >16 triggers, or frac invalid.

    Examples
    --------
    >>> triggers = [
    ...     {'name': 'flat_pt', 'stratify': ['pt_bin'], 'frac': 0.1},
    ...     {'name': 'flat_eta', 'stratify': ['eta_bin'], 'frac': 0.1},
    ... ]
    >>> out = downsampleDFTrigger(df, triggers, random_state=42)
    """
    # --- input validation ---------------------------------------------------
    if len(triggers) > 16:
        raise ValueError(
            f"Maximum 16 triggers supported (uint16 bitmask), got {len(triggers)}"
        )
    required_keys = {"name", "stratify", "frac"}
    for i, t in enumerate(triggers):
        missing_keys = required_keys - set(t.keys())
        if missing_keys:
            raise ValueError(
                f"Trigger {i} missing required keys: {missing_keys}"
            )
        if not (0 < t["frac"] <= 1):
            raise ValueError(
                f"Trigger '{t['name']}': frac must be in (0, 1], got {t['frac']}"
            )

    # --- algorithm (unchanged from pidSkimmedFit.py) ------------------------
    result = df.copy()
    combined_trigger = np.zeros(len(result), dtype=int)
    trigger_bitmask = 1

    for i, trigger in enumerate(triggers):
        name = trigger["name"]
        stratify = trigger["stratify"]
        frac = trigger["frac"]

        if isinstance(stratify, str):
            stratify = [stratify]

        missing = [c for c in stratify if c not in result.columns]
        if missing:
            raise ValueError(
                f"Trigger '{name}': stratify columns not in DataFrame: {missing}"
            )

        group_sizes = result.groupby(stratify).size()
        weights = 1 / group_sizes
        weights /= weights.sum()
        weights = weights.astype(weight_dtype)

        weight_column = f"weight_{name}"
        result = result.merge(
            weights.reset_index(name=weight_column), on=stratify, how="left"
        )

        n_samples = int(len(result) * frac)
        sampled_indices = result.sample(
            n=n_samples, weights=weight_column, replace=False,
            random_state=random_state + i,
        ).index

        combined_trigger[sampled_indices] |= trigger_bitmask
        trigger_bitmask <<= 1

    result["combined_trigger"] = combined_trigger.astype(np.uint16)
    return result[result["combined_trigger"] > 0]


# ===================================================================
# downsampleDFSmoothFactorized — Smooth factorized PDF (v3.1)
# ===================================================================

def downsampleDFSmoothFactorized(
    df: pd.DataFrame,
    frac: float,
    variables: Union[str, List[str]],
    random_state: int,
    n_bins: Union[int, List[int]] = 50,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
    clip_quantile: float = 0.001,
) -> pd.DataFrame:
    """
    Downsample with smooth factorized PDF weighting.

    Estimates a factorized PDF as the product of 1D marginals:

        PDF(x, y, z, ...) ≈ PDF(x) × PDF(y) × PDF(z) × ...

    Each marginal is a 1D histogram with log-space interpolation for empty
    bins, linearly interpolated to give a smooth PDF value at each data
    point.  Sampling weights are proportional to 1/PDF.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    frac : float
        Fraction of rows to retain (0 < frac <= 1).
    variables : str or list of str
        Column(s) to use for PDF estimation.
    random_state : int
        Random seed for reproducibility.
    n_bins : int or list of int, default 50
        Number of histogram bins per variable.
    keep_weights : bool, default True
        If True, include the weight column in the output.
    weight_dtype : np.dtype, default np.float32
        Data type for the weight column.
    weight_column : str, default 'weight'
        Name of the weight column.
    clip_quantile : float, default 0.001
        Quantile for range clipping to avoid sparse tails.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with smooth inverse-PDF weights.

    Raises
    ------
    ValueError
        If frac not in (0, 1], variables missing, or weight_column exists.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'x': np.random.randn(10000), 'y': np.random.randn(10000)})
    >>> out = downsampleDFSmoothFactorized(df, frac=0.1, variables=['x', 'y'],
    ...                                    random_state=42, n_bins=30)
    """
    # --- input validation ---------------------------------------------------
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    if isinstance(variables, str):
        variables = [variables]

    missing = [c for c in variables if c not in df.columns]
    if missing:
        raise ValueError(f"variables not in DataFrame: {missing}")

    if weight_column in df.columns:
        raise ValueError(
            f"weight_column '{weight_column}' already exists in DataFrame"
        )

    if isinstance(n_bins, int):
        n_bins = [n_bins] * len(variables)
    if len(n_bins) != len(variables):
        raise ValueError(
            f"n_bins length ({len(n_bins)}) must match variables length ({len(variables)})"
        )

    # --- estimate factorized PDF: product of 1D marginals -------------------
    log_pdf = np.zeros(len(df), dtype=np.float64)

    for var, nb in zip(variables, n_bins):
        values = df[var].values.astype(np.float64)
        centers, pdf_1d, _ = _estimate_1d_pdf(values, nb, clip_quantile)
        pdf_at_points = np.interp(values, centers, pdf_1d)
        log_pdf += np.log(pdf_at_points)

    # --- weights = 1 / PDF --------------------------------------------------
    log_weights = -log_pdf
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)

    # --- sample -------------------------------------------------------------
    n_samples = int(len(df) * frac)
    return _weighted_sample(
        df, weights, n_samples, random_state,
        keep_weights, weight_column, weight_dtype,
    )


# ===================================================================
# downsampleDFSmooth — Smooth full ND PDF (v3.1)
# ===================================================================

def downsampleDFSmooth(
    df: pd.DataFrame,
    frac: float,
    variables: Union[str, List[str]],
    random_state: int,
    n_bins: Union[int, List[int]] = 50,
    keep_weights: bool = True,
    weight_dtype: np.dtype = np.float32,
    weight_column: str = "weight",
    clip_quantile: float = 0.001,
) -> pd.DataFrame:
    """
    Downsample with smooth full ND PDF weighting.

    Estimates the full joint PDF from an ND histogram with log-space
    interpolation for empty bins, then uses scipy RegularGridInterpolator
    to evaluate the PDF at each data point.

    For D > 3, use downsampleDFSmoothFactorized instead.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. **Not modified.**
    frac : float
        Fraction of rows to retain (0 < frac <= 1).
    variables : str or list of str
        Column(s) to use for PDF estimation.
    random_state : int
        Random seed for reproducibility.
    n_bins : int or list of int, default 50
        Number of histogram bins per variable.
    keep_weights : bool, default True
        If True, include the weight column in the output.
    weight_dtype : np.dtype, default np.float32
        Data type for the weight column.
    weight_column : str, default 'weight'
        Name of the weight column.
    clip_quantile : float, default 0.001
        Quantile for range clipping to avoid sparse tails.

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with smooth inverse-PDF weights.

    Raises
    ------
    ValueError
        If frac not in (0, 1], variables missing, weight_column exists,
        or dimensions > 5.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> df = pd.DataFrame({'x': np.random.randn(10000), 'y': np.random.randn(10000)})
    >>> out = downsampleDFSmooth(df, frac=0.1, variables=['x', 'y'],
    ...                          random_state=42, n_bins=30)
    """
    # --- input validation ---------------------------------------------------
    if not (0 < frac <= 1):
        raise ValueError(f"frac must be in (0, 1], got {frac}")

    if isinstance(variables, str):
        variables = [variables]

    if len(variables) > 5:
        raise ValueError(
            f"downsampleDFSmooth supports up to 5 dimensions, got {len(variables)}. "
            f"Use downsampleDFSmoothFactorized for higher dimensions."
        )

    missing = [c for c in variables if c not in df.columns]
    if missing:
        raise ValueError(f"variables not in DataFrame: {missing}")

    if weight_column in df.columns:
        raise ValueError(
            f"weight_column '{weight_column}' already exists in DataFrame"
        )

    if isinstance(n_bins, int):
        n_bins = [n_bins] * len(variables)
    if len(n_bins) != len(variables):
        raise ValueError(
            f"n_bins length ({len(n_bins)}) must match variables length ({len(variables)})"
        )

    # --- build ND histogram -------------------------------------------------
    data_arrays = [df[v].values.astype(np.float64) for v in variables]

    all_edges = []
    for vals, nb in zip(data_arrays, n_bins):
        lo = np.quantile(vals, clip_quantile)
        hi = np.quantile(vals, 1.0 - clip_quantile)
        all_edges.append(np.linspace(lo, hi, nb + 1))

    hist_nd, _ = np.histogramdd(
        np.column_stack(data_arrays), bins=all_edges
    )

    # Normalise to PDF, then fill empty bins via log-space interpolation
    bin_volumes = np.prod([e[1] - e[0] for e in all_edges])
    pdf_nd = hist_nd / (len(df) * bin_volumes)
    pdf_nd = _interpolate_empty_bins_log_nd(pdf_nd)

    # --- interpolate PDF at each data point ---------------------------------
    from scipy.interpolate import RegularGridInterpolator
    centers = [0.5 * (e[:-1] + e[1:]) for e in all_edges]
    interpolator = RegularGridInterpolator(
        centers, pdf_nd, method="linear",
        bounds_error=False, fill_value=None,
    )

    points = np.column_stack(data_arrays)
    pdf_at_points = interpolator(points)
    pdf_at_points = np.maximum(pdf_at_points, 1e-30)

    # --- weights = 1 / PDF --------------------------------------------------
    log_weights = -np.log(pdf_at_points)
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)

    # --- sample -------------------------------------------------------------
    n_samples = int(len(df) * frac)
    return _weighted_sample(
        df, weights, n_samples, random_state,
        keep_weights, weight_column, weight_dtype,
    )
