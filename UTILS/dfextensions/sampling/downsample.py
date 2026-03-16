"""
dfextensions/sampling/downsample.py

Stratified downsampling utilities for DataFrames.

Six public functions:
  Binned (v3.0 — ported from production):
    - downsampleDF:                   Binned groupby, inverse-group-size weights
    - downsampleDFTrigger:            Multi-trigger bitmask (binned)

  Smooth (v3.1/v3.2):
    - downsampleDFSmoothFactorized:   Smooth PDF, product of 1D marginals
    - downsampleDFSmooth:             Smooth PDF, full ND joint histogram
    - downsampleDFSmoothTrigger:      Multi-trigger bitmask (smooth)

Phase 13.10.DF v3.0 + Phase 13.11.DF
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

    The stored weight column contains the **normalised sampling probability**
    p_i = w_i / sum(w), same convention as downsampleDF.
    To reconstruct the original distribution:
        correction_weight = 1 / p_i
        correction_weight *= N_orig / correction_weight.sum()
    """
    rng = np.random.RandomState(random_state)
    probs = weights / weights.sum()
    n_samples = min(n_samples, len(df))
    chosen = rng.choice(len(df), size=n_samples, replace=False, p=probs)
    result = df.iloc[chosen].copy()
    if keep_weights:
        result[weight_column] = probs[chosen].astype(weight_dtype)
    return result


def _compute_smooth_weights_factorized(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_specs: dict,
) -> np.ndarray:
    """
    Compute inverse-PDF weights using factorized (product of marginals) approach.

    PDF(x, y, cat, ...) = PDF(cat) × PDF_1d(x) × PDF_1d(y) × ...

    Each 1D marginal estimated independently.
    Categorical: value_counts normalized.
    """
    log_pdf = np.zeros(len(df), dtype=np.float64)

    # Categorical marginals
    for col in categorical_cols:
        vc = df[col].value_counts(normalize=True)
        pdf_cat = df[col].map(vc).values.astype(np.float64)
        pdf_cat = np.maximum(pdf_cat, 1e-30)
        log_pdf += np.log(pdf_cat)

    # Continuous marginals
    for col, edges in continuous_specs.items():
        values = df[col].values.astype(np.float64)
        centers, pdf_1d = _estimate_1d_pdf_from_edges(values, edges)
        pdf_at_points = np.interp(values, centers, pdf_1d)
        pdf_at_points = np.maximum(pdf_at_points, 1e-30)
        log_pdf += np.log(pdf_at_points)

    # weights = 1 / PDF
    log_weights = -log_pdf
    log_weights -= log_weights.max()
    return np.exp(log_weights)


def _compute_smooth_weights_nd(
    df: pd.DataFrame,
    categorical_cols: list,
    continuous_specs: dict,
) -> np.ndarray:
    """
    Compute inverse-PDF weights using full ND histogram.

    Per AD-3: one ND histogram. Categorical axes = exact lookup (no interpolation).
    Continuous axes = linear interpolation. Implemented as: groupby categoricals,
    then ND histogram + interpolation on continuous axes per group.
    """
    weights = np.zeros(len(df), dtype=np.float64)
    cont_cols = list(continuous_specs.keys())
    all_edges = [continuous_specs[c] for c in cont_cols]

    if not categorical_cols:
        # Pure continuous: single ND histogram
        weights = _compute_nd_weights_for_group(df, cont_cols, all_edges)
    else:
        # Per AD-3: group by categoricals, ND histogram per group
        grouped = df.groupby(categorical_cols)
        n_groups = grouped.ngroups

        for cat_key, group_idx in grouped.groups.items():
            group_df = df.loc[group_idx]
            if len(group_df) == 0:
                continue

            if not cont_cols:
                # All-categorical per AD-10: uniform within group
                w = np.ones(len(group_df), dtype=np.float64)
            else:
                w = _compute_nd_weights_for_group(group_df, cont_cols, all_edges)

            # Scale by 1/group_fraction (categorical inverse weight)
            group_frac = len(group_df) / len(df)
            w /= group_frac

            weights[df.index.get_indexer(group_idx)] = w

    return weights


def _compute_nd_weights_for_group(
    group_df: pd.DataFrame,
    cont_cols: list,
    all_edges: list,
) -> np.ndarray:
    """Compute inverse-PDF weights for one categorical group on continuous axes."""
    from scipy.interpolate import RegularGridInterpolator

    data_arrays = [group_df[c].values.astype(np.float64) for c in cont_cols]

    if len(cont_cols) == 0:
        return np.ones(len(group_df), dtype=np.float64)

    hist_nd, _ = np.histogramdd(np.column_stack(data_arrays), bins=all_edges)

    # Non-uniform bin volumes
    widths_per_axis = [np.diff(e) for e in all_edges]
    if len(cont_cols) == 1:
        volumes = widths_per_axis[0]
    else:
        volumes = np.prod(
            np.meshgrid(*widths_per_axis, indexing='ij'), axis=0
        )

    pdf_nd = hist_nd / (len(group_df) * volumes)
    pdf_nd = _interpolate_empty_bins_log_nd(pdf_nd)

    centers = [0.5 * (e[:-1] + e[1:]) for e in all_edges]
    interpolator = RegularGridInterpolator(
        centers, pdf_nd, method="linear",
        bounds_error=False, fill_value=None,
    )

    points = np.column_stack(data_arrays)
    if len(cont_cols) == 1:
        points = points.reshape(-1, 1)
    pdf_at_points = interpolator(points)
    pdf_at_points = np.maximum(pdf_at_points, 1e-30)

    log_weights = -np.log(pdf_at_points)
    log_weights -= log_weights.max()
    return np.exp(log_weights)


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
    weights = 1 / group_sizes
    weights /= weights.sum()
    temp_df = df.merge(weights.reset_index(name=weight_column), on=stratify, how="left")
    temp_df[weight_column] = temp_df[weight_column].astype(weight_dtype)
    n_samples = int(len(df) * frac)
    downsampled_df = temp_df.sample(
        n=n_samples, weights=weight_column, replace=False, random_state=random_state,
    )
    if not keep_weights:
        downsampled_df = downsampled_df.drop(columns=[weight_column])
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
# downsampleDFSmoothFactorized — Smooth factorized PDF (v3.2)
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

    Returns
    -------
    pd.DataFrame
        Downsampled DataFrame with inverse-PDF weights.

    Examples
    --------
    >>> variables = {'type': 'categorical', 'pT': (50, 0, 10), 'eta': [-2, -1, 0, 1, 2]}
    >>> out = downsampleDFSmoothFactorized(df, frac=0.1, variables=variables, random_state=42)
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
        )

    weights = _compute_smooth_weights_factorized(work_df, categorical_cols, continuous_specs)
    n_samples = int(len(work_df) * frac)
    return _weighted_sample(
        work_df, weights, n_samples, random_state,
        keep_weights, weight_column, weight_dtype,
    )


# ===================================================================
# downsampleDFSmooth — Smooth full ND PDF (v3.2)
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
        )

    # Reset index for positional weight alignment
    work_df = work_df.reset_index(drop=True)
    weights = _compute_smooth_weights_nd(work_df, categorical_cols, continuous_specs)
    n_samples = int(len(work_df) * frac)
    return _weighted_sample(
        work_df, weights, n_samples, random_state,
        keep_weights, weight_column, weight_dtype,
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
                trigger_df, categorical_cols, continuous_specs
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
