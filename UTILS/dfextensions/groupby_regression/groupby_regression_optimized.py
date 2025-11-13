"""
Optimized GroupByRegressor with improved parallelization for real-world data.

Key improvements:
1. Array-based data passing (reduce serialization overhead)
2. Smart batching for small groups
3. Memory-efficient group processing
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, List, Tuple, Callable, Optional
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, HuberRegressor


# ============================================================================
# SHARED HELPER FUNCTION (Used by V2/V3/V4)
# ============================================================================

def _make_nan_result_row(
    group_key: Union[tuple, int, str],
    gb_columns: List[str],
    fit_columns: List[str],
    linear_columns: List[str],
    suffix: str,
    fit_intercept: bool,
    diag_dict: dict,
    diag_prefix: str = "diag_"
) -> dict:
    """
    Create a result row with NaN fit results and diagnostics.
    
    Used when a group has insufficient data or singular matrix.
    Ensures consistent column structure across V2/V3/V4.
    
    Parameters
    ----------
    group_key : tuple or scalar
        Group identifier(s)
    gb_columns : list[str]
        Group-by column names
    fit_columns : list[str]
        Target variable names
    linear_columns : list[str]
        Predictor variable names
    suffix : str
        Suffix for output columns
    fit_intercept : bool
        Whether intercept was requested
    diag_dict : dict
        Diagnostic values: n_total, n_valid, n_filtered, cond_xtx, status
        Pass empty dict {} when diag=False to suppress diagnostic columns
    diag_prefix : str
        Prefix for diagnostic columns
        
    Returns
    -------
    row : dict
        Dictionary with group keys, NaN fit results, and diagnostics
    """
    # Start with group keys
    if isinstance(group_key, tuple):
        row = dict(zip(gb_columns, group_key))
    else:
        row = {gb_columns[0]: group_key}
    
    # Add NaN fit results for each target
    for target_name in fit_columns:
        # Intercept (only if fit_intercept=True)
        if fit_intercept:
            row[f"{target_name}_intercept{suffix}"] = np.nan
            row[f"{target_name}_intercept_err{suffix}"] = np.nan
        
        # Slopes (always present)
        for predictor_name in linear_columns:
            row[f"{target_name}_slope_{predictor_name}{suffix}"] = np.nan
            row[f"{target_name}_slope_{predictor_name}_err{suffix}"] = np.nan
        
        # RMS (always present)
        row[f"{target_name}_rms{suffix}"] = np.nan
    
    # Add diagnostics (if dict provided)
    if diag_dict:
        row[f"{diag_prefix}n_total"] = diag_dict.get('n_total', 0)
        row[f"{diag_prefix}n_valid"] = diag_dict.get('n_valid', 0)
        row[f"{diag_prefix}n_filtered"] = diag_dict.get('n_filtered', 0)
        row[f"{diag_prefix}cond_xtx"] = diag_dict.get('cond_xtx', np.inf)
        row[f"{diag_prefix}status"] = diag_dict.get('status', 'UNKNOWN')
    
    return row


def process_group_array_based(
    key: tuple,
    indices: np.ndarray,
    X_all: np.ndarray,
    y_all: np.ndarray,
    w_all: np.ndarray,
    gb_columns: List[str],
    target_idx: int,
    predictor_indices: List[int],
    min_stat: int,
    sigmaCut: float,
    fitter: Union[str, Callable],
    max_refits: int = 10,
) -> dict:
    """
    Process a single group using pre-extracted arrays.

    This avoids DataFrame slicing overhead by working directly with NumPy arrays.

    Args:
        key: Group key tuple
        indices: Row indices for this group (into X_all, y_all, w_all)
        X_all: Full predictor array [n_total, n_predictors]
        y_all: Full target array [n_total, n_targets]
        w_all: Full weight array [n_total]
        gb_columns: Group-by column names
        target_idx: Which target column to fit
        predictor_indices: Which predictor columns to use
        min_stat: Minimum rows required
        sigmaCut: Outlier threshold (MAD units)
        fitter: "ols", "robust", or callable
        max_refits: Maximum robust iterations

    Returns:
        Dictionary with fit results for this group
    """
    # Handle single vs multiple group columns
    if isinstance(key, tuple):
        group_dict = dict(zip(gb_columns, key))
    else:
        group_dict = {gb_columns[0]: key}

    if len(indices) < min_stat:
        return group_dict  # Will be filled with NaN by caller

    try:
        # Extract data for this group - single operation, contiguous memory
        X = X_all[indices][:, predictor_indices]
        y = y_all[indices]  # y_all is 1D for single target
        w = w_all[indices]

        # Remove any remaining NaN rows
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(w)
        if valid_mask.sum() < min_stat:
            return group_dict

        X = X[valid_mask]
        y = y[valid_mask]
        w = w[valid_mask]

        # Select fitter
        if callable(fitter):
            model = fitter()
        elif fitter == "ols":
            model = LinearRegression()
        elif fitter == "robust":
            model = HuberRegressor(tol=1e-4)
        else:
            model = LinearRegression()

        # Robust fitting with outlier rejection
        mask = np.ones(len(y), dtype=bool)
        n_refits = 0

        for iteration in range(max_refits):
            if mask.sum() < min_stat:
                break

            X_fit = X[mask]
            y_fit = y[mask]
            w_fit = w[mask]

            # Fit with explicit error handling
            try:
                model.fit(X_fit, y_fit, sample_weight=w_fit)
            except LinAlgError as e:
                # Singular matrix / collinearity
                logging.warning(f"LinAlgError in fit for group {key}: {e}")
                return group_dict  # Return NaNs gracefully
            except Exception as e:
                # Catch any other fitting errors
                logging.warning(f"Unexpected error in fit for group {key}: {e}")
                return group_dict  # Return NaNs gracefully

            # Check for convergence
            if iteration == 0 or sigmaCut > 50:  # No outlier rejection
                break

            # Compute residuals and MAD
            pred = model.predict(X)
            residuals = y - pred
            mad = np.median(np.abs(residuals - np.median(residuals)))

            if mad < 1e-9:  # Perfect fit
                break

            # Update mask
            new_mask = np.abs(residuals) < sigmaCut * mad * 1.4826
            if np.array_equal(mask, new_mask):  # Converged
                break

            mask = new_mask
            n_refits += 1

        # Store results
        group_dict['coefficients'] = model.coef_
        group_dict['intercept'] = model.intercept_
        group_dict['n_refits'] = n_refits
        group_dict['n_used'] = mask.sum()
        group_dict['frac_rejected'] = 1.0 - (mask.sum() / len(y))

        # Compute residual statistics
        pred_final = model.predict(X[mask])
        res_final = y[mask] - pred_final
        group_dict['rms'] = np.sqrt(np.mean(res_final**2))
        group_dict['mad'] = np.median(np.abs(res_final - np.median(res_final))) * 1.4826

    except Exception as e:
        logging.warning(f"Fit failed for group {key}: {e}")

    return group_dict


def process_batch_of_groups(
    batch: List[Tuple[tuple, np.ndarray]],
    X_all: np.ndarray,
    y_all: np.ndarray,
    w_all: np.ndarray,
    gb_columns: List[str],
    target_idx: int,
    predictor_indices: List[int],
    min_stat: int,
    sigmaCut: float,
    fitter: Union[str, Callable],
    max_refits: int,
) -> List[dict]:
    """
    Process multiple small groups in a single worker task.

    This reduces process spawn overhead for datasets with many small groups.
    """
    results = []
    for key, indices in batch:
        result = process_group_array_based(
            key, indices, X_all, y_all, w_all, gb_columns,
            target_idx, predictor_indices, min_stat, sigmaCut, fitter, max_refits
        )
        results.append(result)
    return results


class GroupByRegressorOptimized:
    """
    Optimized version of GroupByRegressor with improved parallelization.
    """

    @staticmethod
    def make_parallel_fit_optimized(
            df: pd.DataFrame,
            gb_columns: List[str],
            fit_columns: List[str],
            linear_columns: List[str],
            median_columns: List[str],
            weights: str,
            suffix: str,
            selection: pd.Series,
            addPrediction: bool = False,
            cast_dtype: Union[str, None] = None,
            n_jobs: int = 1,
            min_stat: Union[int, List[int]] = 10,
            sigmaCut: float = 5.0,
            fitter: Union[str, Callable] = "ols",
            batch_size: Union[str, int] = "auto",
            batch_strategy: str = "auto",
            max_refits: int = 10,
            small_group_threshold: int = 30,
            min_batch_size: int = 10,
            backend: str = 'loky',
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Optimized parallel fitting with array-based data passing and smart batching.
        """
        logger = logging.getLogger(__name__)
        if isinstance(min_stat, list):
            min_stat = min(min_stat) if len(min_stat) > 0 else 1

        # Apply selection
        df_selected = df[selection].copy()
        if df_selected.empty:
            return df.assign(**{f"{col}{suffix}": np.nan for col in fit_columns}), \
                pd.DataFrame(columns=gb_columns)

        # Prepare arrays (array-based path)
        y_matrix = df_selected[fit_columns].to_numpy()
        X_all = df_selected[linear_columns].to_numpy()
        w_all = df_selected[weights].to_numpy() if isinstance(weights, str) else np.ones(len(df_selected))

        # Group indices (array-based)
        grouped = df_selected.groupby(gb_columns, sort=False, observed=True)
        groups_items = list(grouped.groups.items())

        # Choose batching strategy
        def choose_strategy():
            if batch_strategy in ("no_batching", "size_bucketing"):
                return batch_strategy
            # auto
            sizes = np.array([len(idxs) for _, idxs in groups_items])
            if (sizes <= small_group_threshold).mean() > 0.7 and len(groups_items) > 50:
                return "size_bucketing"
            return "no_batching"

        strategy = choose_strategy()

        # Pre-build y index per target
        target_indices = {t: i for i, t in enumerate(fit_columns)}

        target_results: List[Tuple[str, List[dict]]] = []

        for target_col in fit_columns:
            target_idx = target_indices[target_col]

            # batching
            if strategy == "size_bucketing":
                small = [(k, idxs) for k, idxs in groups_items if len(idxs) < small_group_threshold]
                large = [(k, idxs) for k, idxs in groups_items if len(idxs) >= small_group_threshold]

                # Bucket small groups
                small_sorted = sorted(small, key=lambda kv: len(kv[1]), reverse=True)
                buckets: List[List[Tuple[tuple, np.ndarray]]] = []
                current: List[Tuple[tuple, np.ndarray]] = []
                current_size = 0
                for k, idxs in small_sorted:
                    current.append((k, idxs))
                    current_size += len(idxs)
                    if current_size >= max(min_batch_size, small_group_threshold):
                        buckets.append(current)
                        current = []
                        current_size = 0
                if current:
                    buckets.append(current)

                def process_bucket(bucket):
                    out = []
                    for key, idxs in bucket:
                        out.append(process_group_array_based(
                            key, idxs, X_all, y_matrix[:, target_idx], w_all,
                            gb_columns, target_idx, list(range(len(linear_columns))),
                            min_stat, sigmaCut, fitter, max_refits
                        ))
                    return out

                results_small = Parallel(n_jobs=n_jobs, backend=backend)(
                    delayed(process_bucket)(b) for b in buckets
                )
                results_small = [r for sub in results_small for r in sub]

                # Large groups individually
                results_large = Parallel(n_jobs=n_jobs, batch_size=batch_size, backend=backend)(
                    delayed(process_group_array_based)(
                        key, idxs, X_all, y_matrix[:, target_idx], w_all,
                        gb_columns, target_idx, list(range(len(linear_columns))),
                        min_stat, sigmaCut, fitter, max_refits
                    )
                    for key, idxs in large
                )

                results = results_small + results_large

            else:
                # Original approach: each group is a task
                results = Parallel(n_jobs=n_jobs, batch_size=batch_size, backend=backend)(
                    delayed(process_group_array_based)(
                        key, idxs, X_all, y_matrix[:, target_idx], w_all,
                        gb_columns, target_idx, list(range(len(linear_columns))),
                        min_stat, sigmaCut, fitter, max_refits
                    )
                    for key, idxs in groups_items
                )

            target_results.append((target_col, results))

        # Construct dfGB: merge target results horizontally (one row per group)
        dfGB = None
        for t_idx, (target_col, results) in enumerate(target_results):
            df_t = pd.DataFrame(results)
            if df_t.empty:
                continue
            # Expand coefficients into per-predictor columns for this target
            # Expand coefficients into per-predictor columns for this target
            if 'coefficients' in df_t.columns:
                for idx, pred_col in enumerate(linear_columns):
                    colname = f"{target_col}_slope_{pred_col}"
                    df_t[colname] = [
                        (arr[idx] if isinstance(arr, (np.ndarray, list, tuple)) and len(arr) > idx else np.nan)
                        for arr in df_t['coefficients']
                    ]
            if 'intercept' in df_t.columns:
                df_t[f"{target_col}_intercept"] = df_t['intercept']
            if 'rms' in df_t.columns:
                df_t[f"{target_col}_rms"] = df_t['rms']
            if 'mad' in df_t.columns:
                df_t[f"{target_col}_mad"] = df_t['mad']

            # Drop temp columns; for additional targets keep only gb keys + target-specific cols
            drop_cols = ['coefficients', 'intercept', 'rms', 'mad']
            if t_idx > 0:
                keep_cols = set(gb_columns) | {c for c in df_t.columns if c.startswith(f"{target_col}_")}
                df_t = df_t[[c for c in df_t.columns if c in keep_cols]]
            df_t = df_t.drop(columns=[c for c in drop_cols if c in df_t.columns], errors='ignore')

            if dfGB is None:
                dfGB = df_t
            else:
                dfGB = dfGB.merge(df_t, on=gb_columns, how='left')

        if dfGB is None:
            dfGB = pd.DataFrame(columns=gb_columns)

        # Add medians (per-group)
        if median_columns:
            median_results = []
            for key, idxs in grouped.groups.items():
                group_dict = dict(zip(gb_columns, key))
                for col in median_columns:
                    group_dict[col] = df_selected.loc[idxs, col].median()
                median_results.append(group_dict)
            df_medians = pd.DataFrame(median_results)
            dfGB = dfGB.merge(df_medians, on=gb_columns, how='left')

        # Cast dtypes for numeric fit metrics
        if cast_dtype:
            for col in dfGB.columns:
                if any(x in col for x in ['slope', 'intercept', 'rms', 'mad']):
                    dfGB[col] = dfGB[col].astype(cast_dtype)

        # Add suffix (keep gb_columns unchanged)
        dfGB = dfGB.rename(columns={col: f"{col}{suffix}" for col in dfGB.columns if col not in gb_columns})

        # Optionally add predictions back to the input df
        if addPrediction and not dfGB.empty:
            df = df.merge(dfGB, on=gb_columns, how="left")
            for target_col in fit_columns:
                intercept_col = f"{target_col}_intercept{suffix}"
                if intercept_col not in df.columns:
                    continue
                df[f"{target_col}{suffix}"] = df[intercept_col]
                for pred_col in linear_columns:
                    slope_col = f"{target_col}_slope_{pred_col}{suffix}"
                    if slope_col in df.columns:
                        df[f"{target_col}{suffix}"] += df[slope_col] * df[pred_col]

        return df, dfGB



# Convenience wrapper for backward compatibility
def make_parallel_fit_v2(
    df: pd.DataFrame,
    gb_columns: List[str],
    fit_columns: List[str],
    linear_columns: List[str],
    median_columns: List[str],
    weights: str,
    suffix: str,
    selection: pd.Series,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop-in replacement for GroupByRegressor.make_parallel_fit with optimizations.

    Usage:
        # Old way:
        df_out, dfGB = GroupByRegressor.make_parallel_fit(df, ...)

        # New way (same API):
        df_out, dfGB = make_parallel_fit_v2(df, ...)
    """
    return GroupByRegressorOptimized.make_parallel_fit_optimized(
        df, gb_columns, fit_columns, linear_columns, median_columns,
        weights, suffix, selection, **kwargs
    )


# ======================================================================
# Phase 3 – Fast, Vectorized Implementation (NumPy / Numba-ready)
# ======================================================================

import numpy as np
from numpy.linalg import LinAlgError
from numpy.linalg import LinAlgError
import pandas as pd
import time

def make_parallel_fit_v3(
    df: pd.DataFrame,
    *,
    gb_columns: Union[str, List[str]],
    fit_columns: Union[str, List[str]],
    linear_columns: Union[str, List[str]],
    median_columns: Optional[List[str]] = None,
    weights: Optional[str] = None,
    suffix: str = "_fast",
    selection: Optional[pd.Series] = None,
    addPrediction: bool = False,
    fit_intercept: bool = True,  # ← NEW PARAMETER
    cast_dtype: Optional[str] = None,  # ← Changed: don't default to float32
    diag: bool = True,
    diag_prefix: str = "diag_",
    min_stat: Union[int, List[int]] = 3,
):
    """
    Phase 3 – High-performance NumPy implementation with numerical stability.
    
    NEW in V3 v2.0:
    - fit_intercept parameter for regression through origin
    - Inf/NaN filtering before matrix operations
    - Float64 enforcement for stability
    - Enhanced diagnostics (4 status levels)
    - Multi-target support with proper broadcasting
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    gb_columns : str or list[str]
        Columns to group by
    fit_columns : str or list[str]
        Target variable(s) - always returns 2D internally
    linear_columns : str or list[str]
        Predictor variable(s)
    median_columns : list[str], optional
        Columns for per-group medians (not yet implemented)
    weights : str, optional
        Column with sample weights
    suffix : str, default="_fast"
        Suffix for output columns
    selection : pd.Series[bool], optional
        Row mask to select subset
    addPrediction : bool, default=False
        Add fitted predictions to df_out
    fit_intercept : bool, default=True
        If True, fit intercept (normal regression)
        If False, force line through origin (no intercept term)
    cast_dtype : str, optional
        **DEPRECATED in v3.** v3 always uses float64 internally for 
        numerical stability, regardless of input dtype. This parameter
        is kept for backward compatibility but is ignored.
    diag : bool, default=True
        Include diagnostic columns
    diag_prefix : str, default="diag_"
        Prefix for diagnostic columns
    min_stat : int or list[int], default=3
        Minimum number of points per group
        
    Returns
    -------
    df_out : pd.DataFrame
        Copy of input (with predictions if addPrediction=True)
    dfGB : pd.DataFrame
        Per-group fit results with columns:
        - Group keys (from gb_columns)
        - {target}_intercept{suffix} (only if fit_intercept=True)
        - {target}_intercept_err{suffix} (standard error, if fit_intercept=True)
        - {target}_slope_{predictor}{suffix} (always)
        - {target}_slope_{predictor}_err{suffix} (standard error, always)
        - {target}_rms{suffix} (always)
        - diag_n_total (if diag=True)
        - diag_n_valid (if diag=True)
        - diag_n_filtered (if diag=True)
        - diag_cond_xtx (if diag=True)
        - diag_status (if diag=True)
        - diag_time_ms (if diag=True)
        - diag_wall_ms (if diag=True)
                       
    Notes
    -----
    Status levels:
    - 'OK': Normal fit, no numerical issues
    - 'ILL_CONDITIONED_RIDGED': High condition number (>1e12),
                                 small ridge added, result usable
    - 'INSUFFICIENT_DATA': Too few valid points after Inf/NaN filtering
    - 'SINGULAR_MATRIX': Matrix inversion failed despite ridge
    
    Parameter Error Estimates:
    Standard errors are computed analytically using:
        SE(β_i) = sqrt(σ² * [(X'X)^(-1)]_ii)
    where:
        σ² = RMS² (residual variance)
        [(X'X)^(-1)]_ii = i-th diagonal element of inverse covariance matrix
    
    These are asymptotically correct standard errors under OLS assumptions:
    - Linear model
    - Homoscedastic errors (constant variance)
    - Uncorrelated errors
    - Gaussian errors (for exact confidence intervals)
    
    When fit_intercept=False:
    - No centering is applied (preserves "through origin" meaning)
    - Intercept columns are NOT included in output
    - Less numerically stable, but mathematically correct
    
    When diag=False:
    - No diagnostic columns (n_total, n_valid, n_filtered, cond_xtx, status, time)
    - Only fit results (coefficients, errors, RMS) are returned
    - Useful for cleaner output when diagnostics not needed
    
    Examples
    --------
    # Standard regression with intercept
    >>> _, dfGB = make_parallel_fit_v3(
    ...     df=data,
    ...     gb_columns=['sector'],
    ...     fit_columns=['dy', 'dz'],
    ...     linear_columns=['radius'],
    ...     suffix='_fit'
    ... )
    >>> # dfGB has: dy_intercept_fit, dy_slope_radius_fit, dy_rms_fit, ...
    
    # Regression through origin (no intercept)
    >>> _, dfGB = make_parallel_fit_v3(
    ...     df=data,
    ...     gb_columns=['sector'],
    ...     fit_columns=['dy'],
    ...     linear_columns=['radius'],
    ...     fit_intercept=False,
    ...     suffix='_fit'
    ... )
    >>> # dfGB has: dy_slope_radius_fit, dy_rms_fit (NO intercept column)
    """
    
    t_start = time.perf_counter()
    
    # ========================================================================
    # 0. INPUT NORMALIZATION AND VALIDATION
    # ========================================================================
    
    # Normalize column lists
    if isinstance(gb_columns, str):
        gb_columns = [gb_columns]
    else:
        gb_columns = list(gb_columns)
    
    if isinstance(fit_columns, str):
        fit_columns = [fit_columns]
    else:
        fit_columns = list(fit_columns)
    
    if isinstance(linear_columns, str):
        linear_columns = [linear_columns]
    else:
        linear_columns = list(linear_columns)
    
    if median_columns is None:
        median_columns = []
    
    # Handle min_stat
    if isinstance(min_stat, (list, tuple)):
        min_stat = int(np.max(min_stat))
    
    # Apply selection
    if selection is not None:
        df = df.loc[selection]
    
    # Validate we have enough columns
    required_cols = set(gb_columns) | set(fit_columns) | set(linear_columns)
    if weights is not None:
        required_cols.add(weights)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")
    
    # ========================================================================
    # 1. CREATE GROUPS
    # ========================================================================
    
    if len(gb_columns) == 1:
        gb = df.groupby(gb_columns[0], observed=True, sort=False)
    else:
        gb = df.groupby(gb_columns, observed=True, sort=False)
    
    # Count total groups (for progress tracking if needed)
    n_groups = len(gb)
    
    # ========================================================================
    # 2. PREPARE RESULT STORAGE
    # ========================================================================
    
    res_rows = []
    
    # ========================================================================
    # 3. PROCESS EACH GROUP
    # ========================================================================
    
    for g_name, g_df in gb:
        t0 = time.perf_counter()
        
        # Quick check: enough data before extracting arrays
        if len(g_df) < min_stat:
            diag_info = {
                'n_total': len(g_df),
                'n_valid': 0,
                'n_filtered': 0,
                'cond_xtx': np.inf,
                'status': 'INSUFFICIENT_DATA'
            }
            row = _make_nan_result_row(
                g_name, gb_columns, fit_columns, linear_columns,
                suffix, fit_intercept, 
                diag_info if diag else {},  # ← Only pass diag if enabled
                diag_prefix
            )
            res_rows.append(row)
            continue
        
        # ====================================================================
        # 3.1. EXTRACT DATA AS FLOAT64 (CRITICAL FOR STABILITY)
        # ====================================================================
        
        # Extract predictors and targets
        # fit_columns is always a list, so y is always 2D: (n_rows, n_targets)
        X = g_df[linear_columns].to_numpy(dtype=np.float64, copy=False)
        y = g_df[fit_columns].to_numpy(dtype=np.float64, copy=False)
        
        # Shape check (should always be 2D due to list input)
        # X: (n, p) where p = len(linear_columns)
        # y: (n, t) where t = len(fit_columns)
        
        # Extract weights if specified
        if weights is not None:
            w = g_df[weights].to_numpy(dtype=np.float64, copy=False)
        else:
            w = None
        
        # ====================================================================
        # 3.2. FILTER Inf/NaN BEFORE ADDING INTERCEPT
        # ====================================================================
        
        # Build mask: valid = no NaN or Inf in any column
        valid_mask = ~(
            np.isnan(X).any(axis=1) |   # NaN in any predictor
            np.isinf(X).any(axis=1) |   # Inf in any predictor
            np.isnan(y).any(axis=1) |   # NaN in any target
            np.isinf(y).any(axis=1)     # Inf in any target
        )
        
        # Check weights too
        if w is not None:
            valid_mask &= ~(np.isnan(w) | np.isinf(w))
        
        # Count filtering results
        n_total = len(X)
        n_valid = valid_mask.sum()
        n_filtered = n_total - n_valid
        
        # Check if enough valid data remains
        if n_valid < min_stat:
            diag_info = {
                'n_total': n_total,
                'n_valid': n_valid,
                'n_filtered': n_filtered,
                'cond_xtx': np.inf,
                'status': 'INSUFFICIENT_DATA'
            }
            row = _make_nan_result_row(
                g_name, gb_columns, fit_columns, linear_columns,
                suffix, fit_intercept,
                diag_info if diag else {},  # ← Only pass diag if enabled
                diag_prefix
            )
            res_rows.append(row)
            continue
        
        # Apply filter to clean data
        X = X[valid_mask]
        y = y[valid_mask]
        if w is not None:
            w = w[valid_mask]
        
        # ====================================================================
        # 3.3. ADD INTERCEPT COLUMN (IF REQUESTED)
        # ====================================================================
        
        # After filtering, add intercept column if needed
        # This ensures intercept column is always clean (all 1s)
        if fit_intercept:
            X = np.c_[np.ones(len(X)), X]
        # Now X shape: (n_valid, p+1) if intercept, else (n_valid, p)
        
        # ====================================================================
        # 3.4. APPLY WEIGHTS (WEIGHTED LEAST SQUARES)
        # ====================================================================
        
        if w is not None:
            # Transform to weighted problem: X*sqrt(w), y*sqrt(w)
            sw = np.sqrt(w)
            X = X * sw[:, None]
            y = y * sw[:, None]
        
        # ====================================================================
        # 3.5. SOLVE OLS: β = (X'X)^(-1) X'y
        # ====================================================================
        
        try:
            # Compute normal equations
            XtX = X.T @ X
            XtY = X.T @ y
            
            # Check condition number for numerical stability
            cond = np.linalg.cond(XtX)
            
            # Add small ridge if ill-conditioned
            if cond > 1e12:
                # Ridge = small fraction of trace
                ridge = 1e-8 * np.trace(XtX) / len(XtX)
                XtX += ridge * np.eye(len(XtX))
                # Note: We store the ORIGINAL condition number in diagnostics
                # This tells users how ill-conditioned the problem was
                status = 'ILL_CONDITIONED_RIDGED'
            else:
                status = 'OK'
            
            # Solve for coefficients
            beta = np.linalg.solve(XtX, XtY)
            # beta shape: (p+1, t) if intercept, else (p, t)
            
        except np.linalg.LinAlgError as e:
            # Singular matrix - cannot solve even with ridge
            diag_info = {
                'n_total': n_total,
                'n_valid': n_valid,
                'n_filtered': n_filtered,
                'cond_xtx': np.inf,
                'status': f'SINGULAR_MATRIX: {str(e)}'
            }
            row = _make_nan_result_row(
                g_name, gb_columns, fit_columns, linear_columns,
                suffix, fit_intercept,
                diag_info if diag else {},  # ← Only pass diag if enabled
                diag_prefix
            )
            res_rows.append(row)
            continue
        
        # ====================================================================
        # 3.6. COMPUTE PREDICTIONS AND RMS
        # ====================================================================
        
        y_pred = X @ beta
        resid = y - y_pred
        
        # ⭐ CRITICAL FIX: Use degrees of freedom for unbiased variance estimate
        # n_params includes intercept if fit_intercept=True
        n_params = len(beta)  # beta shape: (p+1, t) or (p, t)
        dof = max(n_valid - n_params, 1)  # Avoid division by zero
        
        # Compute unbiased variance estimate: s² = SSR / dof
        # 
        # IMPORTANT: In weighted case, we already applied sqrt(w) to X and y,
        # so resid = sqrt(w) * (y_raw - X_raw @ beta)
        # Therefore: resid² = w * (y_raw - X_raw @ beta)²
        # This is exactly the WLS SSR term we need!
        # 
        # DO NOT multiply by w again - that would give w³ weighting!
        s2 = (resid ** 2).sum(axis=0) / dof
        
        rms = np.sqrt(s2)
        # rms shape: (t,) - one value per target
        # This is in weighted metric, which is correct for covariance formula
        
        # ====================================================================
        # 3.7. COMPUTE PARAMETER ERROR ESTIMATES
        # ====================================================================
        
        # Standard error of parameters: SE(β) = sqrt(σ² * diag((X'X)^(-1)))
        # where σ² = RMS² (residual variance)
        #
        # Derivation:
        # For OLS, Cov(β) = σ² (X'X)^(-1)
        # Standard errors are sqrt of diagonal: SE(β_i) = sqrt(Cov(β)_ii)
        #
        # This gives asymptotically correct standard errors under assumptions:
        # - Errors are homoscedastic (constant variance)
        # - Errors are uncorrelated
        # - Errors are Gaussian (for exact confidence intervals)
        #
        # For weighted least squares, σ² is computed from weighted residuals
        # and (X'X)^(-1) already accounts for weights through weighted X
        
        try:
            # Compute (X'X)^(-1) - we already have XtX
            XtX_inv = np.linalg.inv(XtX)
            
            # Extract diagonal of (X'X)^(-1)
            # This gives variance multiplier for each parameter
            diag_XtX_inv = np.diag(XtX_inv)
            
            # Compute parameter standard errors for each target
            # SE shape: (p+1, t) if intercept, else (p, t)
            # SE[i, j] = sqrt(RMS[j]² * diag_XtX_inv[i])
            param_errors = np.sqrt(
                rms[None, :] ** 2 * diag_XtX_inv[:, None]
            )
            # Broadcasting: (p+1, 1) * (1, t) → (p+1, t)
            
        except np.linalg.LinAlgError:
            # Singular matrix - cannot compute errors
            # This should be rare since we already checked invertibility above
            param_errors = None
        
        # ====================================================================
        # 3.8. STORE RESULTS
        # ====================================================================
        
        t1 = time.perf_counter()
        
        # Build result row
        # Start with group keys
        if isinstance(g_name, tuple):
            row = dict(zip(gb_columns, g_name))
        else:
            row = {gb_columns[0]: g_name}
        
        # Add fit results for each target
        for t_idx, target_name in enumerate(fit_columns):
            
            # Intercept (only if fit_intercept=True)
            if fit_intercept:
                row[f"{target_name}_intercept{suffix}"] = beta[0, t_idx]
                if param_errors is not None:
                    row[f"{target_name}_intercept_err{suffix}"] = param_errors[0, t_idx]
                else:
                    row[f"{target_name}_intercept_err{suffix}"] = np.nan
                slope_start = 1  # Slopes start at index 1
            else:
                slope_start = 0  # Slopes start at index 0
            
            # Slopes (always present)
            for j, predictor_name in enumerate(linear_columns):
                row[f"{target_name}_slope_{predictor_name}{suffix}"] = \
                    beta[slope_start + j, t_idx]
                # Parameter error for this slope
                if param_errors is not None:
                    row[f"{target_name}_slope_{predictor_name}_err{suffix}"] = \
                        param_errors[slope_start + j, t_idx]
                else:
                    row[f"{target_name}_slope_{predictor_name}_err{suffix}"] = np.nan
            
            # RMS (always present)
            row[f"{target_name}_rms{suffix}"] = rms[t_idx]
        
        # Add diagnostics (if enabled)
        if diag:
            row[f"{diag_prefix}n_total"] = n_total
            row[f"{diag_prefix}n_valid"] = n_valid
            row[f"{diag_prefix}n_filtered"] = n_filtered
            row[f"{diag_prefix}cond_xtx"] = cond
            row[f"{diag_prefix}status"] = status
            row[f"{diag_prefix}time_ms"] = (t1 - t0) * 1000
        
        res_rows.append(row)
    
    # ========================================================================
    # 4. ASSEMBLE OUTPUT DATAFRAME
    # ========================================================================
    
    dfGB = pd.DataFrame(res_rows)
    
    # Add wall time diagnostic
    if diag:
        t_end = time.perf_counter()
        dfGB[f"{diag_prefix}wall_ms"] = (t_end - t_start) * 1000
    
    # ========================================================================
    # 5. HANDLE PREDICTIONS (IF REQUESTED)
    # ========================================================================
    
    if addPrediction:
        # TODO: Add predictions to df_out
        # For now, just return copy
        df_out = df.copy()
    else:
        df_out = df.copy()
    
    return df_out, dfGB

def make_parallel_fit_v4(
        *,
        df,
        gb_columns,
        fit_columns,
        linear_columns,
        median_columns=None,
        weights=None,
        suffix="_v4",
        selection=None,
        addPrediction=False,
        cast_dtype="float64",
        min_stat=3,
        diag=False,
        diag_prefix="diag_",
):
    """
    Phase 3 (v4): Numba JIT weighted OLS with *fast* multi-column groupby support.
    Key points:
      - Group boundaries via vectorized adjacent-row comparisons per key column.
      - Vectorized dfGB assembly (no per-group iloc).
    """
    import numpy as np
    import pandas as pd

    if median_columns is None:
        median_columns = []

    # Filter
    if selection is not None:
        df = df.loc[selection]

    # Normalize group columns
    gb_cols = [gb_columns] if isinstance(gb_columns, str) else list(gb_columns)

    # Validate columns
    needed = set(gb_cols) | set(linear_columns) | set(fit_columns)
    if weights is not None:
        needed.add(weights)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Stable sort by all group columns so groups are contiguous
    df_sorted = df.sort_values(gb_cols, kind="mergesort")

    # Dense arrays
    dtype_num = np.float64 if cast_dtype is None else cast_dtype
    X_all = df_sorted[linear_columns].to_numpy(dtype=dtype_num, copy=False)
    Y_all = df_sorted[fit_columns].to_numpy(dtype=dtype_num, copy=False)
    W_all = (np.ones(len(df_sorted), dtype=np.float64) if weights is None
             else df_sorted[weights].to_numpy(dtype=np.float64, copy=False))

    N = X_all.shape[0]
    if N == 0:
        return df_sorted.copy(), pd.DataFrame(columns=gb_cols + [f"n_refits{suffix}", f"n_used{suffix}", f"frac_rejected{suffix}"])

    n_feat = X_all.shape[1]
    n_tgt  = Y_all.shape[1]

    # ---------- FAST multi-column group offsets ----------
    # boundaries[0] = True; boundaries[i] = True if any key column changes at i vs i-1
    boundaries = np.empty(N, dtype=bool)
    boundaries[0] = True
    if N > 1:
        boundaries[1:] = False
        # OR-adjacent compare for each group column (vectorized)
        for col in gb_cols:
            a = df_sorted[col].to_numpy()
            boundaries[1:] |= (a[1:] != a[:-1])

    starts = np.flatnonzero(boundaries)
    offsets = np.empty(len(starts) + 1, dtype=np.int64)
    offsets[:-1] = starts
    offsets[-1] = N
    n_groups = len(starts)
    # ----------------------------------------------------

    # Allocate beta [n_groups, 1+n_feat, n_tgt]
    beta = np.zeros((n_groups, n_feat + 1, n_tgt), dtype=np.float64)

    # Numba kernel (weighted) or NumPy fallback
    try:
        _ols_kernel_numba_weighted(X_all, Y_all, W_all, offsets, n_groups, n_feat, n_tgt, int(min_stat), beta)
    except NameError:
        for gi in range(n_groups):
            i0, i1 = offsets[gi], offsets[gi + 1]
            m = i1 - i0
            if m < int(min_stat):
                continue
            Xg = X_all[i0:i1]
            Yg = Y_all[i0:i1]
            Wg = W_all[i0:i1].reshape(-1)
            X1 = np.c_[np.ones(m), Xg]
            XtX = (X1.T * Wg).dot(X1)
            XtY = (X1.T * Wg).dot(Yg)
            try:
                coeffs = np.linalg.solve(XtX, XtY)
                beta[gi, :, :] = coeffs
            except np.linalg.LinAlgError:
                pass

    # ---------- Vectorized dfGB assembly ----------
    # Pre-take first-row-of-group keys without iloc in a Python loop
    key_arrays = {col: df_sorted[col].to_numpy()[starts] for col in gb_cols}

    # Diagnostics & coeff arrays
    n_refits_arr = np.zeros(n_groups, dtype=np.int32)
    n_used_arr   = (offsets[1:] - offsets[:-1]).astype(np.int32)
    frac_rej_arr = np.zeros(n_groups, dtype=np.float64)

    out_dict = {col: key_arrays[col] for col in gb_cols}
    out_dict[f"n_refits{suffix}"] = n_refits_arr
    out_dict[f"n_used{suffix}"]   = n_used_arr
    out_dict[f"frac_rejected{suffix}"] = frac_rej_arr

    # Intercept + slopes
    for t_idx, tname in enumerate(fit_columns):
        out_dict[f"{tname}_intercept{suffix}"] = beta[:, 0, t_idx].astype(np.float64, copy=False)
        for j, cname in enumerate(linear_columns, start=1):
            out_dict[f"{tname}_slope_{cname}{suffix}"] = beta[:, j, t_idx].astype(np.float64, copy=False)

    # Optional diag: compute in one pass per group
    if diag:
        for t_idx, tname in enumerate(fit_columns):
            rms = np.zeros(n_groups, dtype=np.float64)
            mad = np.zeros(n_groups, dtype=np.float64)
            for gi in range(n_groups):
                i0, i1 = offsets[gi], offsets[gi + 1]
                m = i1 - i0
                if m == 0:
                    continue
                Xg = X_all[i0:i1]
                y  = Y_all[i0:i1, t_idx]
                X1 = np.c_[np.ones(m), Xg]
                resid = y - (X1 @ beta[gi, :, t_idx])
                rms[gi] = np.sqrt(np.mean(resid ** 2))
                mad[gi] = np.median(np.abs(resid - np.median(resid)))
            out_dict[f"{diag_prefix}{tname}_rms{suffix}"] = rms
            out_dict[f"{diag_prefix}{tname}_mad{suffix}"] = mad

    dfGB = pd.DataFrame(out_dict)
    # ---------- end dfGB assembly ----------

    return df_sorted, dfGB
