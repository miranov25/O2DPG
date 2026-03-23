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
import re

# ============================================================================
# PHASE 13.1.GB: PYARROW DETECTION
# ============================================================================

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    _PYARROW_AVAILABLE = True
    _PYARROW_VERSION = tuple(int(x) for x in pa.__version__.split('.')[:2])
except ImportError:
    _PYARROW_AVAILABLE = False
    _PYARROW_VERSION = (0, 0)

# ============================================================================
# PHASE 12.9.GB: NUMBA DETECTION
# ============================================================================

try:
    import numba
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
    _NUMBA_VERSION = tuple(int(x) for x in numba.__version__.split('.')[:2])
except ImportError:
    _NUMBA_AVAILABLE = False
    _NUMBA_VERSION = (0, 0)
    # Dummy decorators for when Numba unavailable
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args else decorator(args[0])
    def prange(*args):
        return range(*args)

# ============================================================================
# PHASE 12.9.GB: STATUS CODES (Numba-compatible integers)
# ============================================================================

STATUS_OK = 0
STATUS_INSUFFICIENT_DATA = 1
STATUS_INSUFFICIENT_VALID = 2
STATUS_ILL_CONDITIONED = 3
STATUS_SINGULAR = 4

STATUS_TO_STRING = {
    STATUS_OK: 'OK',
    STATUS_INSUFFICIENT_DATA: 'INSUFFICIENT_DATA',
    STATUS_INSUFFICIENT_VALID: 'INSUFFICIENT_VALID',
    STATUS_ILL_CONDITIONED: 'ILL_CONDITIONED_RIDGED',
    STATUS_SINGULAR: 'SINGULAR_MATRIX',
}

# Module-level logger
_logger = logging.getLogger(__name__)


# ============================================================================
# PHASE 13.1.GB: PYARROW BACKEND FUNCTIONS
# ============================================================================

def _select_backend(df, backend, threshold):
    """
    Select processing backend for sorting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    backend : str
        'pandas', 'pyarrow', or 'auto'
    threshold : int
        Row count threshold for auto selection
    
    Returns
    -------
    str : 'pandas' or 'pyarrow'
    """
    if backend == 'pyarrow':
        if not _PYARROW_AVAILABLE:
            import warnings
            warnings.warn(
                "PyArrow not available, falling back to pandas. "
                "Install with: pip install pyarrow>=12.0",
                UserWarning
            )
            return 'pandas'
        if _PYARROW_VERSION < (12, 0):
            import warnings
            warnings.warn(
                f"PyArrow {'.'.join(map(str, _PYARROW_VERSION))} < 12.0, falling back to pandas. "
                "Upgrade with: pip install pyarrow>=12.0",
                UserWarning
            )
            return 'pandas'
        return 'pyarrow'
    
    elif backend == 'auto':
        if (_PYARROW_AVAILABLE and 
            _PYARROW_VERSION >= (12, 0) and 
            len(df) > threshold):
            _logger.debug(
                f"Backend auto-selected: pyarrow (rows={len(df)} > threshold={threshold})"
            )
            return 'pyarrow'
        _logger.debug(
            f"Backend auto-selected: pandas (rows={len(df)}, threshold={threshold}, "
            f"pyarrow_available={_PYARROW_AVAILABLE})"
        )
        return 'pandas'
    
    elif backend == 'pandas':
        return 'pandas'
    
    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'pandas', 'pyarrow', or 'auto'")


def _sort_pyarrow(df, needed_cols, gb_cols):
    """
    Memory-efficient sort using PyArrow.
    
    PyArrow sort_indices is stable as of PyArrow 12.0, matching
    pandas mergesort behavior for deterministic group ordering.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    needed_cols : list
        Columns to include in sorted output
    gb_cols : list
        Columns to sort by (group columns)
    
    Returns
    -------
    pa.Table : Sorted PyArrow table
    """
    # Convert only needed columns (memory efficient)
    table = pa.Table.from_pandas(df[needed_cols], preserve_index=False)
    
    # Sort using PyArrow (stable sort as of PyArrow 12.0)
    sort_keys = [(col, 'ascending') for col in gb_cols]
    indices = pc.sort_indices(table, sort_keys=sort_keys)
    sorted_table = table.take(indices)
    
    return sorted_table


def _get_group_boundaries_pyarrow(sorted_table, gb_cols):
    """
    Get group boundary offsets from sorted PyArrow table.
    
    Uses Arrow-native comparisons (pc.not_equal, pc.or_) to avoid
    materializing full column copies. Only the final boolean boundary
    array is converted to numpy.
    
    Parameters
    ----------
    sorted_table : pa.Table
        Sorted PyArrow table
    gb_cols : list
        Group columns
    
    Returns
    -------
    np.ndarray : Group boundary offsets (length = n_groups + 1)
    """
    n = sorted_table.num_rows
    if n == 0:
        return np.array([0], dtype=np.int64)
    if n == 1:
        return np.array([0, 1], dtype=np.int64)
    
    # Compute boundaries using Arrow-native operations (memory efficient)
    # Avoids to_numpy() on full columns which would defeat memory optimization
    boundary_mask_arrow = None
    
    for col in gb_cols:
        arr = sorted_table.column(col)
        
        # Handle chunked arrays (required for slicing)
        if arr.num_chunks > 1:
            arr = arr.combine_chunks()
        
        # Arrow-native comparison: arr[1:] != arr[:-1]
        shifted = arr.slice(1)           # arr[1:]
        original = arr.slice(0, n - 1)   # arr[:-1]
        changed = pc.not_equal(shifted, original)
        
        if boundary_mask_arrow is None:
            boundary_mask_arrow = changed
        else:
            boundary_mask_arrow = pc.or_(boundary_mask_arrow, changed)
    
    # Convert only the boolean boundary array to numpy (small: n-1 elements)
    boundary_changes = boundary_mask_arrow.to_numpy()
    
    # Build full boundary mask
    boundaries = np.zeros(n, dtype=bool)
    boundaries[0] = True  # First row is always a boundary
    boundaries[1:] = boundary_changes
    
    # Convert to offsets: [0, first_boundary, ..., n]
    boundary_indices = np.where(boundaries)[0]
    offsets = np.append(boundary_indices, n).astype(np.int64)
    
    return offsets


def _extract_arrays_pyarrow(sorted_table, linear_cols, fit_cols, weights_col):
    """
    Extract numpy arrays from PyArrow table.
    
    Uses zero-copy where possible for memory efficiency.
    
    Parameters
    ----------
    sorted_table : pa.Table
        Sorted PyArrow table
    linear_cols : list
        Predictor column names
    fit_cols : list
        Target column names
    weights_col : str or None
        Weight column name
    
    Returns
    -------
    tuple : (X_all, Y_all, W_all) as numpy arrays
    """
    # Extract predictors
    if len(linear_cols) == 1:
        X_all = sorted_table.column(linear_cols[0]).to_numpy().reshape(-1, 1)
    else:
        X_all = np.column_stack([
            sorted_table.column(col).to_numpy() for col in linear_cols
        ])
    
    # Extract targets
    if len(fit_cols) == 1:
        Y_all = sorted_table.column(fit_cols[0]).to_numpy().reshape(-1, 1)
    else:
        Y_all = np.column_stack([
            sorted_table.column(col).to_numpy() for col in fit_cols
        ])
    
    # Extract weights
    if weights_col is not None:
        W_all = sorted_table.column(weights_col).to_numpy()
    else:
        W_all = np.ones(sorted_table.num_rows, dtype=np.float64)
    
    # Ensure float64 for numerical stability
    X_all = X_all.astype(np.float64, copy=False)
    Y_all = Y_all.astype(np.float64, copy=False)
    W_all = W_all.astype(np.float64, copy=False)
    
    return X_all, Y_all, W_all


# ============================================================================
# PHASE 12.4a: METADATA EXPORT FUNCTIONS
# ============================================================================

_SAFE_NAME_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')


def _validate_column_names(columns, context=""):
    """
    Validate that column names are safe for formula generation.
    
    Raises ValueError if any column name contains unsafe characters.
    """
    for name in columns:
        if not _SAFE_NAME_PATTERN.match(name):
            raise ValueError(
                f"Column name '{name}' is not safe for formula generation{context}. "
                f"Use only letters, digits, and underscores, starting with letter or underscore."
            )


def _build_prediction_formula(target, linear_columns, suffix, fit_intercept):
    """
    Generate prediction formula string.
    
    Examples:
        >>> _build_prediction_formula('dyC2', ['rrel', 'rrel2'], '_Fit', True)
        'dyC2_intercept_Fit + dyC2_slope_rrel_Fit*rrel + dyC2_slope_rrel2_Fit*rrel2'
    
    Raises:
        ValueError: If fit_intercept=False and linear_columns is empty
                    (would produce invalid empty formula)
    """
    terms = []
    
    if fit_intercept:
        terms.append(f"{target}_intercept{suffix}")
    
    for col in linear_columns:
        terms.append(f"{target}_slope_{col}{suffix}*{col}")
    
    if not terms:
        raise ValueError(
            f"Cannot build prediction formula for '{target}': "
            f"fit_intercept=False and no linear_columns provided. "
            f"At least one predictor or intercept is required."
        )
    
    return " + ".join(terms)


def _build_residual_formula(target, prediction_formula):
    """Generate residual formula: target - prediction."""
    return f"{target} - ({prediction_formula})"


def _build_pull_formula(residual_formula, error_column):
    """Generate pull formula: residual / error."""
    return f"({residual_formula}) / {error_column}"


def _preprocess_linear_columns(
        df: pd.DataFrame,
        linear_columns: list,
) -> tuple:
    """Normalize linear_columns: evaluate expression tuples, return augmented DataFrame.

    Phase 13.13.GB: Supports (key_name, expression) tuples alongside plain column names.

    Returns (df_augmented, normalized_columns, column_map).
    """
    df_columns = set(df.columns)
    normalized = []
    new_columns = {}
    column_map = {}
    seen_keys = set()

    for item in linear_columns:
        if isinstance(item, str):
            if item not in df_columns:
                raise ValueError(
                    f"Column '{item}' not found in DataFrame. "
                    f"For expressions, use tuple: (key_name, expression)")
            if item in seen_keys:
                raise ValueError(f"Duplicate linear column: '{item}'")
            seen_keys.add(item)
            normalized.append(item)
            column_map[item] = item
        elif isinstance(item, tuple) and len(item) == 2:
            key_name, expression = item
            if not isinstance(key_name, str) or not key_name.isidentifier():
                raise ValueError(f"Key name '{key_name}' is not a valid Python identifier")
            if key_name in df_columns:
                raise ValueError(
                    f"Key name '{key_name}' conflicts with existing DataFrame column. "
                    f"Choose a different key name.")
            if key_name in seen_keys:
                raise ValueError(f"Duplicate key name: '{key_name}'")
            seen_keys.add(key_name)
            try:
                new_columns[key_name] = df.eval(expression)
            except Exception as e:
                raise ValueError(
                    f"Failed to evaluate expression '{expression}' "
                    f"for key '{key_name}': {e}") from e
            normalized.append(key_name)
            column_map[key_name] = expression
        else:
            raise TypeError(
                f"linear_columns must be str or (key_name, expression) tuple, "
                f"got {type(item)}: {item!r}")

    if new_columns:
        df_augmented = df.assign(**new_columns)
    else:
        df_augmented = df

    return df_augmented, normalized, column_map


def _build_fit_metadata(
    fit_columns,
    linear_columns,
    suffix,
    fit_intercept,
    gb_columns,
    weights_column=None,
    median_columns=None,
    min_stat=3,
    diag=False,
    diag_prefix="diag_",
    fit_type="linear",
):
    """
    Build complete metadata dict for fit results.
    
    Returns dict with formulas, column categorizations, and parameters.
    """
    # Validate column names (strict - raises on invalid)
    _validate_column_names(fit_columns, " in fit_columns")
    _validate_column_names(linear_columns, " in linear_columns")
    
    # Initialize metadata structure
    metadata = {
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
            'diagnostics': [],
            'medians': [],
        },
        'parameters': {
            'suffix': suffix,
            'fit_intercept': fit_intercept,
            'min_stat': min_stat,
            'fit_type': fit_type,
            'weights_column': weights_column,
            'pull_default': 'rms',
        },
    }
    
    # Build formulas for each target
    for target in fit_columns:
        # Prediction formula
        pred_formula = _build_prediction_formula(
            target, linear_columns, suffix, fit_intercept
        )
        
        # Residual formula
        resid_formula = _build_residual_formula(target, pred_formula)
        
        # Pull formulas (both RMS and MAD based)
        rms_col = f"{target}_rms{suffix}"
        mad_col = f"{target}_mad{suffix}"
        
        pull_rms = _build_pull_formula(resid_formula, rms_col)
        pull_mad = _build_pull_formula(resid_formula, f"({mad_col} * 1.4826)")
        
        # Store formulas
        metadata['formulas'][f"{target}_pred{suffix}"] = pred_formula
        metadata['residual_formulas'][f"{target}_delta{suffix}"] = resid_formula
        metadata['pull_formulas'][f"{target}_pull{suffix}"] = pull_rms
        metadata['pull_formulas'][f"{target}_pull_mad{suffix}"] = pull_mad
        
        # Build coefficient column names
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
        metadata['columns']['quality'][target] = [rms_col, mad_col]
    
    # Median columns (if provided)
    if median_columns:
        metadata['columns']['medians'] = [f"{col}{suffix}" for col in median_columns]
    
    # Diagnostic columns (if enabled)
    if diag:
        metadata['columns']['diagnostics'] = [
            f"{diag_prefix}n_total{suffix}",
            f"{diag_prefix}n_valid{suffix}",
            f"{diag_prefix}n_filtered{suffix}",
            f"{diag_prefix}cond_xtx{suffix}",
            f"{diag_prefix}status{suffix}",
        ]
    
    return metadata


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
        row[f"{diag_prefix}n_total{suffix}"] = diag_dict.get('n_total', 0)
        row[f"{diag_prefix}n_valid{suffix}"] = diag_dict.get('n_valid', 0)
        row[f"{diag_prefix}n_filtered{suffix}"] = diag_dict.get('n_filtered', 0)
        row[f"{diag_prefix}cond_xtx{suffix}"] = diag_dict.get('cond_xtx', np.inf)
        row[f"{diag_prefix}status{suffix}"] = diag_dict.get('status', 'UNKNOWN')
    
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
    return_metadata: bool = False,
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
    return_metadata : bool, default=False
        If True, return a third value containing metadata with auto-generated
        formulas for predictions, residuals, and pulls.
        
    Returns
    -------
    If return_metadata=False (default):
        df_out : pd.DataFrame
            DataFrame containing ONLY columns needed for fitting:
            gb_columns, fit_columns, linear_columns, median_columns, and weights.
            Note: Extra columns from input are not preserved (Phase 12.5.GB memory optimization).
        dfGB : pd.DataFrame
            Per-group fit results
    If return_metadata=True:
        df_out : pd.DataFrame
        dfGB : pd.DataFrame
        metadata : dict
            Auto-generated formulas and column categorizations with keys:
            - version: Schema version ('1.0')
            - formulas: Prediction formula expressions
            - residual_formulas: Delta formula expressions
            - pull_formulas: Pull formula expressions (RMS and MAD based)
            - columns: Categorized output column names
            - parameters: Fit settings for reproducibility
    
    dfGB columns:
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
    if median_columns:
        required_cols.update(median_columns)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in DataFrame: {missing}")
    
    # === Phase 12.5.GB: Memory optimization (parity with V4) ===
    # Select only needed columns before groupby to reduce memory
    # Build list with deterministic order (no set randomization)
    _needed_cols = gb_columns + fit_columns + linear_columns + (median_columns or [])
    if weights is not None:
        _needed_cols = _needed_cols + [weights]
    _needed_cols = list(dict.fromkeys(_needed_cols))  # Dedupe preserving order
    df = df[_needed_cols]
    # === End Phase 12.5.GB ===
    
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
        # Save unweighted data for MAD computation
        X_unweighted = X.copy()
        y_unweighted = y.copy()
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
        # 3.6b. COMPUTE MAD (Median Absolute Deviation)
        # ====================================================================

        # MAD is robust alternative to RMS, less sensitive to outliers
        # MAD is ALWAYS computed from UNWEIGHTED residuals
        y_pred_unweighted = X_unweighted @ beta
        resid_unweighted = y_unweighted - y_pred_unweighted

        # Compute MAD for each target: median(|resid - median(resid)|)
        n_targets = y_unweighted.shape[1]  # ✅ Get from array shape
        mad = np.zeros(n_targets)
        for t_idx in range(n_targets):
            resid_t = resid_unweighted[:, t_idx]
            mad[t_idx] = np.median(np.abs(resid_t - np.median(resid_t)))
        # mad shape: (t,) - one value per target

        # ====================================================================
        # 3.7. COMPUTE PARAMETER ERROR ESTIMATES
        
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

            # MAD (always present)
            row[f"{target_name}_mad{suffix}"] = mad[t_idx]

        # Add diagnostics (if enabled)
        if diag:
            row[f"{diag_prefix}n_total{suffix}"] = n_total
            row[f"{diag_prefix}n_valid{suffix}"] = n_valid
            row[f"{diag_prefix}n_filtered{suffix}"] = n_filtered
            row[f"{diag_prefix}cond_xtx{suffix}"] = cond
            row[f"{diag_prefix}status{suffix}"] = status
            row[f"{diag_prefix}time_ms{suffix}"] = (t1 - t0) * 1000
        
        res_rows.append(row)
    
    # ========================================================================
    # 4. ASSEMBLE OUTPUT DATAFRAME
    # ========================================================================
    
    dfGB = pd.DataFrame(res_rows)
    
    # Add wall time diagnostic
    if diag:
        t_end = time.perf_counter()
        dfGB[f"{diag_prefix}wall_ms{suffix}"] = (t_end - t_start) * 1000
    
    # ========================================================================
    # 5. HANDLE PREDICTIONS (IF REQUESTED)
    # ========================================================================
    
    if addPrediction:
        # TODO: Add predictions to df_out
        # For now, just return copy
        df_out = df.copy()
    else:
        df_out = df.copy()
    
    # ========================================================================
    # 6. BUILD METADATA (IF REQUESTED)
    # ========================================================================
    
    if return_metadata:
        # Get the actual min_stat value used (may have been converted from list)
        min_stat_val = min_stat if isinstance(min_stat, int) else int(np.max(min_stat))
        
        metadata = _build_fit_metadata(
            fit_columns=fit_columns,
            linear_columns=linear_columns,
            suffix=suffix,
            fit_intercept=fit_intercept,
            gb_columns=gb_columns,
            weights_column=weights,
            median_columns=median_columns,
            min_stat=min_stat_val,
            diag=diag,
            diag_prefix=diag_prefix,
            fit_type='linear_v3',
        )
        return df_out, dfGB, metadata
    
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
        fit_intercept: bool = True,
        cast_dtype: str = "float64",
        min_stat=3,
        diag=False,
        diag_prefix="diag_",
        return_metadata: bool = False,
        # Phase 13.1.GB: PyArrow backend parameters
        backend: str = 'auto',
        pyarrow_threshold: int = 1_000_000,
):
    """
    Phase 3 (v4): Numba JIT weighted OLS with fast multi-column groupby support.
    
    NEW in V4 v2.0 (Phase 2 - November 2025):
    - fit_intercept parameter for regression through origin
    - Parameter error estimates for all coefficients
    - RMS with degrees of freedom correction
    - Inf/NaN filtering before computation
    - Enhanced diagnostics with condition numbers
    - Parity with V3 enhancements
    
    NEW in V4 v3.0 (Phase 13.1.GB - December 2025):
    - Optional PyArrow backend for memory-efficient sorting
    - Reduced memory fragmentation for large datasets
    
    Key features:
    - Group boundaries via vectorized adjacent-row comparisons per key column
    - Vectorized dfGB assembly (no per-group iloc)
    - Optional Numba JIT acceleration (falls back to NumPy if unavailable)
    - Inf/NaN filtering with diagnostics
    - PyArrow backend for memory-efficient sorting (optional)
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    gb_columns : str or list[str]
        Columns to group by
    fit_columns : str or list[str]
        Target variable(s)
    linear_columns : str or list[str]
        Predictor variable(s)
    median_columns : list[str], optional
        Columns for per-group medians (not yet implemented)
    weights : str, optional
        Column with sample weights
    suffix : str, default="_v4"
        Suffix for output columns
    selection : pd.Series[bool], optional
        Row mask to select subset
    addPrediction : bool, default=False
        Add fitted predictions to df_out
    fit_intercept : bool, default=True
        If True, fit intercept (normal regression)
        If False, force line through origin (no intercept term)
    cast_dtype : str, default="float64"
        Data type for computation (always float64 internally)
    min_stat : int, default=3
        Minimum number of points per group
    diag : bool, default=False
        Include diagnostic columns
    diag_prefix : str, default="diag_"
        Prefix for diagnostic columns
    return_metadata : bool, default=False
        If True, return a third value containing metadata with auto-generated
        formulas for predictions, residuals, and pulls.
    backend : str, default='auto'
        Memory backend for sorting.
        - 'pandas': Standard pandas implementation (always works)
        - 'pyarrow': Use PyArrow for memory-efficient sorting (requires pyarrow>=12.0)
        - 'auto': Use PyArrow if available and len(df) > pyarrow_threshold
        
        Note: PyArrow is used for sorting/storage only. Computation remains in
        NumPy/Numba. This reduces memory fragmentation for large datasets.
    pyarrow_threshold : int, default=1_000_000
        Minimum row count to trigger automatic PyArrow backend selection.
        Only applies when backend='auto'. This is a pilot heuristic and
        may be adjusted in future versions.
        
    Returns
    -------
    If return_metadata=False (default):
        df_out : pd.DataFrame
            DataFrame containing ONLY columns needed for fitting:
            gb_columns, fit_columns, linear_columns, median_columns, and weights.
            Note: Extra columns from input are not preserved (Phase 12.5.GB memory optimization).
        dfGB : pd.DataFrame
            Per-group fit results
    If return_metadata=True:
        df_out : pd.DataFrame
        dfGB : pd.DataFrame  
        metadata : dict
            Auto-generated formulas and column categorizations
    
    dfGB columns:
        - Group keys (from gb_columns)
        - {target}_intercept{suffix} (only if fit_intercept=True)
        - {target}_intercept_err{suffix} (standard error, if fit_intercept=True)
        - {target}_slope_{predictor}{suffix} (always)
        - {target}_slope_{predictor}_err{suffix} (standard error, always)
        - {target}_rms{suffix} (RMS with dof correction)
        - {target}_mad{suffix} (MAD)
        - diag_* columns (if diag=True)
        
    Notes
    -----
    This implementation matches V3 functionality but uses vectorized operations
    and optional Numba JIT compilation for performance.
    
    Error estimates use the same formula as V3:
        SE(β_i) = sqrt(σ² * [(X'X)^(-1)]_ii)
    where σ² = SSR / (n - p) with degrees of freedom correction.
    
    Phase 13.1.GB: When backend='pyarrow' or auto-selected, sorting uses PyArrow
    for reduced memory fragmentation. This is particularly beneficial for large
    datasets (>1M rows) on batch farms with limited memory per core.
    """
    import numpy as np
    import pandas as pd

    # Normalize min_stat (V4 parity with V3 - accept list input)
    if isinstance(min_stat, (list, tuple)):
        min_stat = int(np.max(min_stat))

    if median_columns is None:
        median_columns = []

    # Normalize group columns
    gb_cols = [gb_columns] if isinstance(gb_columns, str) else list(gb_columns)
    fit_cols = [fit_columns] if isinstance(fit_columns, str) else list(fit_columns)
    linear_cols = [linear_columns] if isinstance(linear_columns, str) else list(linear_columns)

    # Preprocess linear_columns: evaluate expression tuples (Phase 13.13.GB)
    _has_expressions = any(isinstance(item, tuple) for item in linear_cols)
    if _has_expressions:
        df, linear_cols, _linear_col_map = _preprocess_linear_columns(df, linear_cols)
    else:
        _linear_col_map = {c: c for c in linear_cols}

    # Validate columns (including median_columns) - Phase 12.5.GB
    needed = set(gb_cols) | set(linear_cols) | set(fit_cols)
    if weights is not None:
        needed.add(weights)
    if median_columns:
        needed.update(median_columns)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # === Phase 12.5.GB + 13.1.GB: Memory optimization ===
    # Build column list with deterministic order (no set randomization)
    _needed_cols = gb_cols + fit_cols + linear_cols + (median_columns or [])
    if weights is not None:
        _needed_cols = _needed_cols + [weights]
    _needed_cols = list(dict.fromkeys(_needed_cols))  # Dedupe preserving order
    
    # Apply selection filter BEFORE column subset
    if selection is not None:
        df = df.loc[selection]
    
    # Phase 13.1.GB: Select backend
    use_backend = _select_backend(df, backend, pyarrow_threshold)
    
    if use_backend == 'pyarrow':
        # === PyArrow path (memory efficient) ===
        sorted_table = _sort_pyarrow(df, _needed_cols, gb_cols)
        X_all, Y_all, W_all = _extract_arrays_pyarrow(
            sorted_table, linear_cols, fit_cols, weights
        )
        # Get group boundaries without converting to pandas
        offsets = _get_group_boundaries_pyarrow(sorted_table, gb_cols)
        n_groups = len(offsets) - 1
        N = sorted_table.num_rows
        # Keep sorted_table for later df_sorted creation (lazy)
        _sorted_table = sorted_table
        df_sorted = None  # Will create if needed at return
    else:
        # === Pandas path (existing code) ===
        _sorted_table = None
        # Subset columns first (Phase 12.5.GB memory optimization)
        df = df[_needed_cols]
        # Stable sort by all group columns so groups are contiguous
        df_sorted = df.sort_values(gb_cols, kind="mergesort")
        
        # Dense arrays (always float64 for stability)
        dtype_num = np.float64
        X_all = df_sorted[linear_cols].to_numpy(dtype=dtype_num, copy=False)
        Y_all = df_sorted[fit_cols].to_numpy(dtype=dtype_num, copy=False)
        W_all = (np.ones(len(df_sorted), dtype=np.float64) if weights is None
                 else df_sorted[weights].to_numpy(dtype=np.float64, copy=False))
        
        N = X_all.shape[0]
        
        # Compute group boundaries (pandas path)
        boundaries = np.empty(N, dtype=bool)
        boundaries[0] = True
        if N > 1:
            boundaries[1:] = False
            for col in gb_cols:
                a = df_sorted[col].to_numpy()
                boundaries[1:] |= (a[1:] != a[:-1])
        
        starts = np.flatnonzero(boundaries)
        offsets = np.empty(len(starts) + 1, dtype=np.int64)
        offsets[:-1] = starts
        offsets[-1] = N
        n_groups = len(starts)
    # === End Phase 13.1.GB ===

    if N == 0:
        empty_cols = gb_cols + [f"n_refits{suffix}", f"n_used{suffix}", f"frac_rejected{suffix}"]
        if _sorted_table is not None:
            df_sorted = _sorted_table.to_pandas()
        return df_sorted.copy() if df_sorted is not None else pd.DataFrame(), pd.DataFrame(columns=empty_cols)

    n_feat = X_all.shape[1]
    n_tgt  = Y_all.shape[1]
    
    # Number of parameters (intercept + slopes)
    n_params = (1 + n_feat) if fit_intercept else n_feat

    # ========================================================================
    # ALLOCATE OUTPUT ARRAYS
    # ========================================================================
    
    # beta: [n_groups, n_params, n_tgt]
    beta = np.full((n_groups, n_params, n_tgt), np.nan, dtype=np.float64)
    
    # errors: [n_groups, n_params, n_tgt]
    errors = np.full((n_groups, n_params, n_tgt), np.nan, dtype=np.float64)

    # rms: [n_groups, n_tgt]
    rms_arr = np.full((n_groups, n_tgt), np.nan, dtype=np.float64)

    # mad: [n_groups, n_tgt]
    mad_arr = np.full((n_groups, n_tgt), np.nan, dtype=np.float64)

    # Diagnostics
    n_total_arr = np.zeros(n_groups, dtype=np.int32)
    n_valid_arr = np.zeros(n_groups, dtype=np.int32)
    n_filtered_arr = np.zeros(n_groups, dtype=np.int32)
    cond_arr = np.full(n_groups, np.inf, dtype=np.float64)
    status_arr = np.array(['UNKNOWN'] * n_groups, dtype=object)

    # ========================================================================
    # PROCESS EACH GROUP
    # ========================================================================
    
    # NumPy fallback (Numba kernel would be similar but JIT-compiled)
    for gi in range(n_groups):
        i0, i1 = offsets[gi], offsets[gi + 1]
        m = i1 - i0
        
        n_total_arr[gi] = m
        
        # Quick check: enough data before extracting
        if m < int(min_stat):
            n_valid_arr[gi] = 0
            n_filtered_arr[gi] = 0
            status_arr[gi] = 'INSUFFICIENT_DATA'
            continue
        
        # Extract data for this group
        Xg = X_all[i0:i1]  # (m, n_feat)
        Yg = Y_all[i0:i1]  # (m, n_tgt)
        Wg = W_all[i0:i1]  # (m,)
        
        # ====================================================================
        # FILTER Inf/NaN
        # ====================================================================
        
        # Build valid mask
        valid_mask = ~(
            np.isnan(Xg).any(axis=1) |
            np.isinf(Xg).any(axis=1) |
            np.isnan(Yg).any(axis=1) |
            np.isinf(Yg).any(axis=1) |
            np.isnan(Wg) |
            np.isinf(Wg)
        )
        
        n_valid = valid_mask.sum()
        n_filtered = m - n_valid
        
        n_valid_arr[gi] = n_valid
        n_filtered_arr[gi] = n_filtered
        
        # Check if enough valid data remains
        if n_valid < int(min_stat):
            status_arr[gi] = 'INSUFFICIENT_DATA'
            continue
        
        # Apply filter
        Xg = Xg[valid_mask]
        Yg = Yg[valid_mask]
        Wg = Wg[valid_mask]
        
        # ====================================================================
        # BUILD DESIGN MATRIX
        # ====================================================================
        
        if fit_intercept:
            X1 = np.c_[np.ones(n_valid), Xg]  # (n_valid, 1+n_feat)
        else:
            X1 = Xg  # (n_valid, n_feat)
        
        # ====================================================================
        # APPLY WEIGHTS
        # ====================================================================
        
        # Transform to weighted problem
        sw = np.sqrt(Wg)
        X_weighted = X1 * sw[:, None]
        Y_weighted = Yg * sw[:, None]
        
        # ====================================================================
        # SOLVE OLS
        # ====================================================================
        
        try:
            XtX = X_weighted.T @ X_weighted
            XtY = X_weighted.T @ Y_weighted
            
            # Check condition number
            cond = np.linalg.cond(XtX)
            cond_arr[gi] = cond
            
            # Add ridge if ill-conditioned
            if cond > 1e12:
                ridge = 1e-8 * np.trace(XtX) / len(XtX)
                XtX += ridge * np.eye(len(XtX))
                status_arr[gi] = 'ILL_CONDITIONED_RIDGED'
            else:
                status_arr[gi] = 'OK'
            
            # Solve
            coeffs = np.linalg.solve(XtX, XtY)  # (n_params, n_tgt)
            beta[gi, :, :] = coeffs
            
            # ================================================================
            # COMPUTE RMS (with dof correction)
            # ================================================================
            
            y_pred = X_weighted @ coeffs
            resid = Y_weighted - y_pred
            
            dof = max(n_valid - n_params, 1)
            
            # resid already includes sqrt(w), so resid² = w * e²
            s2 = (resid ** 2).sum(axis=0) / dof  # (n_tgt,)
            rms_arr[gi, :] = np.sqrt(s2)

            # ================================================================
            # COMPUTE MAD (Median Absolute Deviation)
            # ================================================================

            # MAD from unweighted residuals (robust metric)
            y_pred_unweighted = X1 @ coeffs  # X1 is unweighted design matrix
            resid_unweighted = Yg - y_pred_unweighted

            # Compute MAD for each target
            for t_idx in range(n_tgt):
                resid_t = resid_unweighted[:, t_idx]
                mad_val = np.median(np.abs(resid_t - np.median(resid_t)))
                mad_arr[gi, t_idx] = mad_val

            # ================================================================
            # COMPUTE PARAMETER ERRORS
            # ================================================================
            
            try:
                XtX_inv = np.linalg.inv(XtX)
                diag_XtX_inv = np.diag(XtX_inv)
                
                # errors[gi]: (n_params, n_tgt)
                # Broadcasting: (n_params, 1) * (1, n_tgt) → (n_params, n_tgt)
                errors[gi, :, :] = np.sqrt(s2[None, :] * diag_XtX_inv[:, None])
                
            except np.linalg.LinAlgError:
                # Singular - errors remain NaN
                pass
            
        except np.linalg.LinAlgError as e:
            status_arr[gi] = f'SINGULAR_MATRIX'
            continue

    # ========================================================================
    # VECTORIZED OUTPUT ASSEMBLY
    # ========================================================================
    
    # Group start indices
    starts = offsets[:-1]
    
    # Pre-take first-row-of-group keys
    # Phase 13.1.GB: Handle both pandas and PyArrow paths
    if _sorted_table is not None:
        # PyArrow path: extract from Arrow table
        key_arrays = {}
        for col in gb_cols:
            arr = _sorted_table.column(col)
            if arr.num_chunks > 1:
                arr = arr.combine_chunks()
            key_arrays[col] = arr.to_numpy()[starts]
    else:
        # Pandas path: extract from DataFrame
        key_arrays = {col: df_sorted[col].to_numpy()[starts] for col in gb_cols}
    
    out_dict = {col: key_arrays[col] for col in gb_cols}
    
    # Add fit results for each target
    for t_idx, tname in enumerate(fit_cols):
        
        # Intercept (only if fit_intercept=True)
        if fit_intercept:
            out_dict[f"{tname}_intercept{suffix}"] = beta[:, 0, t_idx]
            out_dict[f"{tname}_intercept_err{suffix}"] = errors[:, 0, t_idx]
            slope_start = 1
        else:
            slope_start = 0
        
        # Slopes (always present)
        for j, cname in enumerate(linear_cols):
            out_dict[f"{tname}_slope_{cname}{suffix}"] = beta[:, slope_start + j, t_idx]
            out_dict[f"{tname}_slope_{cname}_err{suffix}"] = errors[:, slope_start + j, t_idx]
        # Add RMS to diagnostics
        for t_idx, tname in enumerate(fit_cols):
            out_dict[f"{tname}_rms{suffix}"] = rms_arr[:, t_idx]
            out_dict[f"{tname}_mad{suffix}"] = mad_arr[:, t_idx]
    # Diagnostics (if enabled)
    if diag:
        out_dict[f"{diag_prefix}n_total{suffix}"] = n_total_arr
        out_dict[f"{diag_prefix}n_valid{suffix}"] = n_valid_arr
        out_dict[f"{diag_prefix}n_filtered{suffix}"] = n_filtered_arr
        out_dict[f"{diag_prefix}cond_xtx{suffix}"] = cond_arr
        out_dict[f"{diag_prefix}status{suffix}"] = status_arr



    dfGB = pd.DataFrame(out_dict)

    # ========================================================================
    # BUILD METADATA (IF REQUESTED)
    # ========================================================================
    
    if return_metadata:
        # Get the actual min_stat value used (may have been converted from list)
        min_stat_val = min_stat if isinstance(min_stat, int) else int(np.max(min_stat))
        
        metadata = _build_fit_metadata(
            fit_columns=fit_cols,
            linear_columns=linear_cols,
            suffix=suffix,
            fit_intercept=fit_intercept,
            gb_columns=gb_cols,
            weights_column=weights,
            median_columns=median_columns if median_columns else None,
            min_stat=min_stat_val,
            diag=diag,
            diag_prefix=diag_prefix,
            fit_type='linear_v4',
        )
        if _has_expressions:
            metadata['linear_column_map'] = _linear_col_map
            metadata['linear_columns_normalized'] = list(linear_cols)
        # Phase 13.1.GB: Create df_sorted from PyArrow table if needed
        if _sorted_table is not None and df_sorted is None:
            df_sorted = _sorted_table.to_pandas()
        return df_sorted, dfGB, metadata

    # Phase 13.1.GB: Create df_sorted from PyArrow table if needed
    if _sorted_table is not None and df_sorted is None:
        df_sorted = _sorted_table.to_pandas()
    return df_sorted, dfGB


# ============================================================================
# PHASE 12.8.GB: BATCH FITTING (V5)
# ============================================================================

def _compute_sort_indices_v5(df, gb_columns, backend='auto'):
    """
    Compute permutation indices for multi-key stable sort.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data
    gb_columns : list[str]
        Columns to sort by (in order of priority)
    backend : str
        'pyarrow', 'pandas', or 'auto'
        
    Returns
    -------
    perm : np.ndarray[int64]
        Indices that would sort df by gb_columns (stable sort)
    """
    gb_cols = [gb_columns] if isinstance(gb_columns, str) else list(gb_columns)
    
    use_pyarrow = (
        backend == 'pyarrow' or 
        (backend == 'auto' and _PYARROW_AVAILABLE)
    )
    
    if use_pyarrow and _PYARROW_AVAILABLE:
        # PyArrow multi-key sort (stable)
        table = pa.Table.from_pandas(df[gb_cols])
        perm = pc.sort_indices(
            table,
            sort_keys=[(col, "ascending") for col in gb_cols]
        ).to_numpy()
    else:
        # NumPy fallback: lexsort uses REVERSED column order
        keys = [df[col].values for col in reversed(gb_cols)]
        perm = np.lexsort(keys)
    
    return perm.astype(np.int64)


def _compute_group_boundaries_v5(df, gb_columns, perm):
    """
    Compute group boundary offsets from sorted permutation.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input data (original, unsorted)
    gb_columns : list[str]
        Groupby columns
    perm : np.ndarray
        Sort permutation indices
        
    Returns
    -------
    offsets : np.ndarray[int64]
        Array of shape (n_groups + 1,) with group boundaries.
        offsets[i] = start of group i in sorted order
        offsets[-1] = len(df)
        
    Notes
    -----
    NaN semantics: NaN != NaN in numpy, which would incorrectly split
    NaN-groups into singletons. gb_columns should not contain NaN/null.
    """
    N = len(perm)
    if N == 0:
        return np.array([0], dtype=np.int64)
    if N == 1:
        return np.array([0, 1], dtype=np.int64)
    
    gb_cols = [gb_columns] if isinstance(gb_columns, str) else list(gb_columns)
    
    # Compute boundaries by comparing adjacent rows in sorted order
    boundaries = np.zeros(N, dtype=bool)
    boundaries[0] = True
    
    for col in gb_cols:
        col_vals = df[col].values
        sorted_vals = col_vals[perm]
        boundaries[1:] |= (sorted_vals[1:] != sorted_vals[:-1])
    
    # Convert to offsets
    starts = np.flatnonzero(boundaries)
    offsets = np.empty(len(starts) + 1, dtype=np.int64)
    offsets[:-1] = starts
    offsets[-1] = N
    
    return offsets


def _compute_chunk_boundaries_v5(n_groups, n_chunks, offsets):
    """
    Compute chunk boundaries that never split groups.
    
    Parameters
    ----------
    n_groups : int
        Total number of groups
    n_chunks : int
        Number of chunks
    offsets : np.ndarray
        Group boundary offsets (length n_groups + 1)
        
    Returns
    -------
    list of tuples: (g_start, g_end, row_start, row_end, local_offsets)
    
    Invariant
    ---------
    Chunks are defined over group indices and therefore never split a group.
    """
    # Ceiling division for even distribution
    groups_per_chunk = (n_groups + n_chunks - 1) // n_chunks
    
    chunk_specs = []
    for chunk_idx in range(n_chunks):
        g_start = chunk_idx * groups_per_chunk
        g_end = min((chunk_idx + 1) * groups_per_chunk, n_groups)
        
        if g_start >= n_groups:
            break  # No more groups
        
        # Row boundaries DERIVED from group offsets
        row_start = offsets[g_start]
        row_end = offsets[g_end]
        
        # Local offsets (relative to row_start)
        local_offsets = offsets[g_start:g_end + 1] - row_start
        
        chunk_specs.append((g_start, g_end, int(row_start), int(row_end), local_offsets))
    
    return chunk_specs


def _validate_v5_params(fit_columns, suffixes, linear_columns, weights):
    """
    Validate and normalize batch fit parameters.
    
    Returns normalized (suffixes, linear_columns, weights) as lists.
    """
    n_fits = len(fit_columns)
    
    # === Suffixes validation ===
    if isinstance(suffixes, str):
        if len(fit_columns) != len(set(fit_columns)):
            duplicates = [x for x in fit_columns if fit_columns.count(x) > 1]
            raise ValueError(
                f"Shared suffix '{suffixes}' not allowed when fit_columns "
                f"contains duplicates: {set(duplicates)}. "
                f"Provide list of suffixes to distinguish output columns."
            )
        suffixes = [suffixes] * n_fits
    else:
        suffixes = list(suffixes)
        if len(suffixes) != n_fits:
            raise ValueError(
                f"suffixes length ({len(suffixes)}) != fit_columns length ({n_fits})"
            )
        # Check for duplicate (target, suffix) pairs
        pairs = list(zip(fit_columns, suffixes))
        if len(pairs) != len(set(pairs)):
            raise ValueError(f"Duplicate (fit_column, suffix) pairs: {pairs}")
    
    # === linear_columns validation ===
    if not linear_columns:
        raise ValueError("linear_columns cannot be empty")
    
    if isinstance(linear_columns[0], str):
        # Shared across all fits
        linear_columns = [list(linear_columns)] * n_fits
    else:
        linear_columns = [list(lc) for lc in linear_columns]
        if len(linear_columns) != n_fits:
            raise ValueError(
                f"linear_columns length ({len(linear_columns)}) != "
                f"fit_columns length ({n_fits})"
            )
    
    # === weights validation ===
    if weights is None:
        weights = [None] * n_fits
    elif isinstance(weights, str):
        weights = [weights] * n_fits
    else:
        weights = list(weights)
        if len(weights) != n_fits:
            raise ValueError(
                f"weights length ({len(weights)}) != fit_columns length ({n_fits})"
            )
    
    return suffixes, linear_columns, weights


def _set_threading_policy_v5(n_jobs):
    """
    Set BLAS threading policy to prevent oversubscription.
    
    Returns dict of original values for restoration.
    """
    import os
    
    env_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    
    original = {}
    for var in env_vars:
        original[var] = os.environ.get(var)
        os.environ[var] = "1"
    
    return original


def _restore_threading_policy_v5(original):
    """Restore original threading environment variables."""
    import os
    
    for var, val in original.items():
        if val is not None:
            os.environ[var] = val
        else:
            os.environ.pop(var, None)


def make_parallel_fit_v5(
    *,
    df,
    gb_columns,
    
    # === Batch Fit Specification ===
    fit_columns,
    suffixes="_v5",
    linear_columns,
    weights=None,
    
    # === Common Parameters ===
    selection=None,
    fit_intercept=True,
    min_stat=3,
    compute_mad=True,
    
    # === Output Control ===
    diag=False,
    diag_prefix="diag_",
    return_metadata=False,
    
    # === Performance ===
    n_jobs=1,
    n_chunks=None,
    backend='auto',
    parallel_backend='auto',
):
    """
    Batch OLS fitting: multiple fits with same groupby in single pass.
    
    Performs weighted ordinary least squares regression for multiple 
    target/predictor combinations. All fits share the same groupby,
    enabling significant optimization over calling v4 multiple times.
    
    Phase 12.8.GB: This function processes data in chunks defined by
    group boundaries, reducing peak memory from O(N) to O(N/chunks).
    
    Phase 12.9.GB: Added Numba parallel processing within chunks.
    
    Physics Context
    ---------------
    In track fitting and calibration workflows:
    
    - **Intercept (offset):** Constant bias at reference position
    - **Slopes:** Linear dependence on position variables
    
    Example distortion model:
        dy = offset_y + slope_rrel * rrel + slope_rrel2 * rrel²
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame. Pass the full DataFrame — the function extracts
        needed columns internally. Avoid pre-slicing.
        
        **Precision:** All computations use float64 for numerical stability.
        This ensures consistent results regardless of input dtype.
        
    gb_columns : str or list[str]
        Columns defining groups. Each unique combination gets separate fits.
        Must not contain null/NaN values (ValueError raised otherwise).
        Example: ["track_index", "firstTForbit"]
        
    fit_columns : list[str]
        Target variables, one per fit. Can contain duplicates when fitting
        same target with different parameters.
        Example: ["dyC2", "dzC2", "dzC2"]
        
    suffixes : str or list[str], default="_v5"
        Suffix(es) for output column names.
        - str: Same suffix for all fits (only if fit_columns unique)
        - list[str]: Per-fit suffix (required if fit_columns has duplicates)
        
        Raises ValueError if would create duplicate output columns.
        
    linear_columns : list[str] or list[list[str]]
        Predictor variables.
        - list[str]: Same predictors for all fits
        - list[list[str]]: Different predictors per fit
        
    weights : str or list[str], optional
        Weight column(s) for weighted least squares.
        - None: Unweighted (all weights = 1)
        - str: Same weights for all fits
        - list[str]: Different weights per fit
        
    selection : array-like of bool, optional
        Row mask applied before grouping.
        
    fit_intercept : bool, default=True
        If True, include intercept term.
        
    min_stat : int, default=3
        Minimum points per group for valid fit.
        
    compute_mad : bool, default=True
        Compute Median Absolute Deviation. Set False for ~30% speedup.
        
    diag : bool, default=False
        Include diagnostic columns.
        
    diag_prefix : str, default="diag_"
        Prefix for diagnostic columns.
        
    return_metadata : bool, default=False
        If True, return (dfGB, metadata) with auto-generated formulas.
        
    n_jobs : int, default=1
        Parallel workers for group processing.
        - 1: Sequential
        - N: Use N workers  
        - -1: Use all CPU cores
        
        Note: When n_jobs > 1, OMP/MKL/OPENBLAS_NUM_THREADS are set to 1
        internally to prevent BLAS oversubscription with Numba prange.
        
    n_chunks : int, optional
        Number of data chunks for memory efficiency.
        Default: max(n_jobs, 1).
        Higher = less memory, same CPU time.
        
        Chunks are defined over group indices and therefore 
        **never split a group**.
        
    backend : str, default='auto'
        Sort backend: 'pandas', 'pyarrow', or 'auto'.
        
    parallel_backend : str, default='auto'
        Parallel processing backend (Phase 12.9.GB).
        - 'auto': Use Numba if available and n_jobs > 1, else sequential
        - 'numba': Force Numba (raises ImportError if unavailable)
        - 'sequential': No parallelization (12.8.GB behavior)
        
    Returns
    -------
    dfGB : pd.DataFrame
        Per-group fit results (merged for all fits).
        
    metadata : dict (only if return_metadata=True)
        Per-fit specifications and auto-generated formulas.
        
    Raises
    ------
    ValueError
        If suffixes would create duplicate output columns.
        If list parameters have inconsistent lengths.
        If gb_columns contain null/NaN values.
        
    KeyError
        If required columns not found in df.
        
    Notes
    -----
    Memory efficiency: Data is processed in chunks defined by group
    boundaries. Only one chunk is materialized at a time, reducing 
    peak memory from O(N) to O(N/chunks).
    
    Numerical precision: All computations use float64 for numerical
    stability, regardless of input dtype. Results match v4 within 
    float64 roundoff.
    
    Threading: When n_jobs > 1, the function sets OMP/MKL/OPENBLAS 
    thread env vars to 1 to prevent oversubscription with Numba prange.
    
    Diagnostics (Phase 12.9.GB): When diag=True:
    - Shared: diag_n_total_v5 (rows per group, same for all fits)
    - Per-fit: diag_n_valid{suffix}, diag_n_filtered{suffix}, 
      diag_cond{suffix}, diag_status{suffix}
    Different fits may have different valid row counts if their 
    targets/weights/predictors contain different NaN patterns.
    
    Examples
    --------
    Basic usage:
    
    >>> dfGB = make_parallel_fit_v5(
    ...     df=data,
    ...     gb_columns=["track_index", "firstTForbit"],
    ...     fit_columns=["dyC2", "dzC2"],
    ...     linear_columns=["rrel", "rrel2"],
    ...     suffixes="_Fit",
    ... )
    
    Different weights per fit:
    
    >>> dfGB = make_parallel_fit_v5(
    ...     df=data,
    ...     gb_columns=["track_index", "firstTForbit"],
    ...     fit_columns=["dzC2", "dzC2"],
    ...     suffixes=["_TPC", "_ITS"],
    ...     linear_columns=["rrel"],
    ...     weights=["weightTPCR", "weightITSR"],
    ...     n_jobs=4,
    ... )
    
    Full calibration workflow:
    
    >>> dfGB, meta = make_parallel_fit_v5(
    ...     df=aDF.df,
    ...     gb_columns=["track_index", "firstTForbit"],
    ...     selection=isOKTrackFitTPCITS,
    ...     fit_columns=["dyC2", "dzC2", "dzC2"],
    ...     suffixes=["_Y", "_Z", "_ZITS"],
    ...     linear_columns=[["rrel", "rrel2"], ["rrel"], ["rrel"]],
    ...     weights=["weightTPCR", "weightTPCR", "weightITSR"],
    ...     return_metadata=True,
    ...     n_jobs=4,
    ... )
    >>> print(meta["fits"]["_Y"]["formulas"]["prediction"])
    
    See Also
    --------
    make_parallel_fit_v4 : Single-fit version
    """
    import os
    
    # ========================================================================
    # 1. VALIDATE & PARSE PARAMETERS
    # ========================================================================
    
    # Normalize gb_columns
    gb_cols = [gb_columns] if isinstance(gb_columns, str) else list(gb_columns)
    
    # Normalize fit_columns
    fit_cols = list(fit_columns)
    n_fits = len(fit_cols)
    
    if n_fits == 0:
        raise ValueError("fit_columns cannot be empty")
    
    # Validate and normalize batch parameters
    suffixes_list, linear_cols_list, weights_list = _validate_v5_params(
        fit_cols, suffixes, linear_columns, weights
    )
    
    # Collect all unique columns needed
    unique_cols = set(gb_cols)
    unique_cols.update(fit_cols)
    for lc in linear_cols_list:
        unique_cols.update(lc)
    for w in weights_list:
        if w is not None:
            unique_cols.add(w)
    
    # Check all columns exist
    missing = [c for c in unique_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    
    # Check for null/NaN in gb_columns
    for col in gb_cols:
        if df[col].isna().any():
            raise ValueError(
                f"gb_columns must not contain null/NaN. Column '{col}' has null values."
            )
    
    # Apply selection
    if selection is not None:
        df = df.loc[selection]
    
    N = len(df)
    if N == 0:
        # Return empty DataFrame with correct columns
        return _build_empty_v5_result(
            gb_cols, fit_cols, suffixes_list, linear_cols_list, 
            fit_intercept, compute_mad, diag, diag_prefix, return_metadata
        )
    
    # ========================================================================
    # 2. COMPUTE SORT ORDER (once)
    # ========================================================================
    
    perm = _compute_sort_indices_v5(df, gb_cols, backend)
    offsets = _compute_group_boundaries_v5(df, gb_cols, perm)
    n_groups = len(offsets) - 1
    
    # ========================================================================
    # 3. SETUP CHUNKING
    # ========================================================================
    
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 4
    
    if n_chunks is None:
        n_chunks = max(n_jobs, 1)
    
    chunk_specs = _compute_chunk_boundaries_v5(n_groups, n_chunks, offsets)
    
    # ========================================================================
    # 4. ALLOCATE OUTPUT ARRAYS
    # ========================================================================
    
    # Determine max params across fits
    n_params_list = []
    for fi in range(n_fits):
        n_lin = len(linear_cols_list[fi])
        n_p = (1 + n_lin) if fit_intercept else n_lin
        n_params_list.append(n_p)
    max_params = max(n_params_list)
    
    # Output arrays (float64)
    beta = np.full((n_groups, n_fits, max_params), np.nan, dtype=np.float64)
    errors = np.full((n_groups, n_fits, max_params), np.nan, dtype=np.float64)
    rms_arr = np.full((n_groups, n_fits), np.nan, dtype=np.float64)
    mad_arr = np.full((n_groups, n_fits), np.nan, dtype=np.float64)
    
    # Diagnostics - shared (1D)
    n_total_arr = np.zeros(n_groups, dtype=np.int32)
    
    # Diagnostics - per-fit (2D) - Phase 12.9.GB
    n_valid_arr = np.zeros((n_groups, n_fits), dtype=np.int32)
    n_filtered_arr = np.zeros((n_groups, n_fits), dtype=np.int32)
    cond_arr = np.full((n_groups, n_fits), np.nan, dtype=np.float64)
    
    # Status array: int for Numba, string for sequential
    selected_backend = _select_parallel_backend(parallel_backend, n_jobs)
    if selected_backend == 'numba':
        status_arr = np.zeros((n_groups, n_fits), dtype=np.int32)
    else:
        status_arr = np.empty((n_groups, n_fits), dtype='U32')
        status_arr[:] = ''
    
    # ========================================================================
    # 5. SET THREADING POLICY
    # ========================================================================
    
    original_threading = None
    if n_jobs > 1:
        original_threading = _set_threading_policy_v5(n_jobs)
        if _NUMBA_AVAILABLE:
            try:
                numba.set_num_threads(n_jobs)
            except Exception:
                pass
    
    # ========================================================================
    # 6. PROCESS CHUNKS
    # ========================================================================
    
    try:
        for chunk_spec in chunk_specs:
            g_start, g_end, row_start, row_end, local_offsets = chunk_spec
            n_groups_chunk = g_end - g_start
            
            if n_groups_chunk == 0:
                continue
            
            # Materialize chunk data (preserve dtype)
            chunk_perm = perm[row_start:row_end]
            chunk_data = {}
            for col in unique_cols:
                chunk_data[col] = df[col].values[chunk_perm]
            
            # Dispatch to appropriate backend (Phase 12.9.GB)
            if selected_backend == 'numba':
                # Prepare dense matrices for Numba kernel
                (X_mat, Y_mat, W_mat,
                 fit_target_idx, fit_weight_idx, fit_n_params_arr,
                 linear_col_indices, linear_col_starts) = _prepare_chunk_matrices_for_numba(
                    chunk_data, fit_cols, linear_cols_list, weights_list, n_params_list
                )
                
                # Call Numba kernel
                _process_chunk_numba(
                    X_mat, Y_mat, W_mat,
                    local_offsets.astype(np.int64),
                    fit_target_idx, fit_weight_idx, fit_n_params_arr,
                    linear_col_indices, linear_col_starts,
                    fit_intercept,
                    beta, errors, rms_arr, mad_arr,
                    n_total_arr, n_valid_arr, n_filtered_arr, cond_arr, status_arr,
                    g_start,
                    min_stat, compute_mad,
                )
            else:
                # Sequential Python fallback
                _process_chunk_v5(
                    chunk_data=chunk_data,
                    local_offsets=local_offsets,
                    g_start=g_start,
                    n_groups_chunk=n_groups_chunk,
                    n_fits=n_fits,
                    fit_cols=fit_cols,
                    linear_cols_list=linear_cols_list,
                    weights_list=weights_list,
                    n_params_list=n_params_list,
                    fit_intercept=fit_intercept,
                    min_stat=min_stat,
                    compute_mad=compute_mad,
                    # Output arrays
                    beta=beta,
                    errors=errors,
                    rms_arr=rms_arr,
                    mad_arr=mad_arr,
                    n_total_arr=n_total_arr,
                    n_valid_arr=n_valid_arr,
                    n_filtered_arr=n_filtered_arr,
                    cond_arr=cond_arr,
                    status_arr=status_arr,
                )
            
            # Free chunk memory
            del chunk_data
            del chunk_perm
    
    finally:
        # Restore threading policy
        if original_threading is not None:
            _restore_threading_policy_v5(original_threading)
    
    # Convert status codes to strings if using Numba backend
    if selected_backend == 'numba':
        status_str_arr = np.empty((n_groups, n_fits), dtype='U32')
        for gi in range(n_groups):
            for fi in range(n_fits):
                status_str_arr[gi, fi] = STATUS_TO_STRING.get(status_arr[gi, fi], '')
        status_arr = status_str_arr
    
    # ========================================================================
    # 7. BUILD OUTPUT DATAFRAME
    # ========================================================================
    
    # Get group keys
    out_dict = {}
    for col in gb_cols:
        sorted_vals = df[col].values[perm]
        out_dict[col] = sorted_vals[offsets[:-1]]
    
    # Add fit results for each fit
    for fi in range(n_fits):
        target = fit_cols[fi]
        suffix = suffixes_list[fi]
        lin_cols = linear_cols_list[fi]
        n_p = n_params_list[fi]
        
        # Coefficients
        if fit_intercept:
            out_dict[f"{target}_intercept{suffix}"] = beta[:, fi, 0]
            out_dict[f"{target}_intercept_err{suffix}"] = errors[:, fi, 0]
            for j, lc in enumerate(lin_cols):
                out_dict[f"{target}_slope_{lc}{suffix}"] = beta[:, fi, j + 1]
                out_dict[f"{target}_slope_{lc}_err{suffix}"] = errors[:, fi, j + 1]
        else:
            for j, lc in enumerate(lin_cols):
                out_dict[f"{target}_slope_{lc}{suffix}"] = beta[:, fi, j]
                out_dict[f"{target}_slope_{lc}_err{suffix}"] = errors[:, fi, j]
        
        # Quality metrics
        out_dict[f"{target}_rms{suffix}"] = rms_arr[:, fi]
        out_dict[f"{target}_mad{suffix}"] = mad_arr[:, fi]
    
    # Add diagnostics
    if diag:
        # Shared diagnostics (1D)
        out_dict[f"{diag_prefix}n_total_v5"] = n_total_arr
        
        # Per-fit diagnostics (Phase 12.9.GB)
        for fi in range(n_fits):
            suffix = suffixes_list[fi]
            out_dict[f"{diag_prefix}n_valid{suffix}"] = n_valid_arr[:, fi]
            out_dict[f"{diag_prefix}n_filtered{suffix}"] = n_filtered_arr[:, fi]
            out_dict[f"{diag_prefix}cond{suffix}"] = cond_arr[:, fi]
            out_dict[f"{diag_prefix}status{suffix}"] = status_arr[:, fi]
    
    dfGB = pd.DataFrame(out_dict)
    
    # ========================================================================
    # 8. BUILD METADATA (if requested)
    # ========================================================================
    
    if return_metadata:
        metadata = _build_v5_metadata(
            gb_cols=gb_cols,
            fit_cols=fit_cols,
            suffixes_list=suffixes_list,
            linear_cols_list=linear_cols_list,
            weights_list=weights_list,
            fit_intercept=fit_intercept,
            min_stat=min_stat,
            compute_mad=compute_mad,
            n_groups=n_groups,
        )
        return dfGB, metadata
    
    return dfGB


def _process_chunk_v5(
    chunk_data,
    local_offsets,
    g_start,
    n_groups_chunk,
    n_fits,
    fit_cols,
    linear_cols_list,
    weights_list,
    n_params_list,
    fit_intercept,
    min_stat,
    compute_mad,
    # Output arrays
    beta,
    errors,
    rms_arr,
    mad_arr,
    n_total_arr,
    n_valid_arr,      # Now 2D: [n_groups, n_fits]
    n_filtered_arr,   # Now 2D: [n_groups, n_fits]
    cond_arr,
    status_arr,
):
    """
    Process all groups in a chunk, performing all fits for each group.
    
    Sequential version. For parallel processing, use _process_chunk_numba
    with parallel_backend='numba'.
    
    Phase 12.9.GB: n_valid_arr and n_filtered_arr are now per-fit (2D).
    """
    for gi_local in range(n_groups_chunk):
        gi_global = g_start + gi_local
        i0 = local_offsets[gi_local]
        i1 = local_offsets[gi_local + 1]
        m = i1 - i0  # Number of rows in this group
        
        n_total_arr[gi_global] = m
        
        if m < min_stat:
            for fi in range(n_fits):
                status_arr[gi_global, fi] = STATUS_TO_STRING[STATUS_INSUFFICIENT_DATA]
            continue
        
        # Process each fit for this group
        for fi in range(n_fits):
            target = fit_cols[fi]
            lin_cols = linear_cols_list[fi]
            weight_col = weights_list[fi]
            n_params = n_params_list[fi]
            
            # Extract data for this group and fit
            Yg = chunk_data[target][i0:i1].astype(np.float64)
            
            # Build design matrix
            n_lin = len(lin_cols)
            if fit_intercept:
                Xg = np.ones((m, n_params), dtype=np.float64)
                for j, lc in enumerate(lin_cols):
                    Xg[:, j + 1] = chunk_data[lc][i0:i1].astype(np.float64)
            else:
                Xg = np.empty((m, n_params), dtype=np.float64)
                for j, lc in enumerate(lin_cols):
                    Xg[:, j] = chunk_data[lc][i0:i1].astype(np.float64)
            
            # Get weights
            if weight_col is not None:
                Wg = chunk_data[weight_col][i0:i1].astype(np.float64)
            else:
                Wg = np.ones(m, dtype=np.float64)
            
            # Filter invalid values
            valid_mask = np.isfinite(Yg) & np.isfinite(Wg) & (Wg > 0)
            for j in range(n_params):
                valid_mask &= np.isfinite(Xg[:, j])
            
            n_valid = np.sum(valid_mask)
            n_filtered = m - n_valid
            
            # Per-fit diagnostics (Phase 12.9.GB)
            n_valid_arr[gi_global, fi] = n_valid
            n_filtered_arr[gi_global, fi] = n_filtered
            
            if n_valid < min_stat:
                status_arr[gi_global, fi] = STATUS_TO_STRING[STATUS_INSUFFICIENT_VALID]
                continue
            
            # Apply filter
            Xg = Xg[valid_mask]
            Yg = Yg[valid_mask]
            Wg = Wg[valid_mask]
            
            # Weighted design matrix
            sqrt_w = np.sqrt(Wg)
            X_weighted = Xg * sqrt_w[:, None]
            Y_weighted = Yg * sqrt_w
            
            try:
                # Normal equations
                XtX = X_weighted.T @ X_weighted
                XtY = X_weighted.T @ Y_weighted
                
                # Check condition number
                cond = np.linalg.cond(XtX)
                cond_arr[gi_global, fi] = cond
                
                # Add ridge if ill-conditioned
                if cond > 1e12:
                    ridge = 1e-8 * np.trace(XtX) / len(XtX)
                    XtX += ridge * np.eye(len(XtX))
                    status_arr[gi_global, fi] = STATUS_TO_STRING[STATUS_ILL_CONDITIONED]
                else:
                    status_arr[gi_global, fi] = STATUS_TO_STRING[STATUS_OK]
                
                # Solve
                coeffs = np.linalg.solve(XtX, XtY)
                beta[gi_global, fi, :n_params] = coeffs
                
                # Compute RMS
                y_pred = X_weighted @ coeffs
                resid = Y_weighted - y_pred
                dof = n_valid - n_params
                if dof > 0:
                    s2 = np.sum(resid ** 2) / dof
                    rms_arr[gi_global, fi] = np.sqrt(s2)
                    
                    # Compute parameter errors
                    try:
                        XtX_inv = np.linalg.inv(XtX)
                        errors[gi_global, fi, :n_params] = np.sqrt(s2 * np.diag(XtX_inv))
                    except np.linalg.LinAlgError:
                        pass
                
                # Compute MAD (if enabled)
                if compute_mad:
                    # Use unweighted residuals for MAD
                    y_pred_uw = Xg @ coeffs
                    resid_uw = Yg - y_pred_uw
                    sorted_resid = np.sort(resid_uw)
                    n_r = len(sorted_resid)
                    if n_r % 2 == 1:
                        med = sorted_resid[n_r // 2]
                    else:
                        med = (sorted_resid[n_r // 2 - 1] + sorted_resid[n_r // 2]) / 2.0
                    abs_dev = np.abs(resid_uw - med)
                    sorted_dev = np.sort(abs_dev)
                    if n_r % 2 == 1:
                        mad_arr[gi_global, fi] = sorted_dev[n_r // 2]
                    else:
                        mad_arr[gi_global, fi] = (sorted_dev[n_r // 2 - 1] + sorted_dev[n_r // 2]) / 2.0
                
            except np.linalg.LinAlgError:
                status_arr[gi_global, fi] = STATUS_TO_STRING[STATUS_SINGULAR]
                continue


# ============================================================================
# PHASE 12.9.GB: NUMBA PARALLEL KERNEL
# ============================================================================

def _select_parallel_backend(parallel_backend, n_jobs):
    """
    Select the parallel processing backend.
    
    Parameters
    ----------
    parallel_backend : str
        'auto', 'numba', or 'sequential'
    n_jobs : int
        Number of parallel jobs requested
        
    Returns
    -------
    str : 'numba' or 'sequential'
    """
    if parallel_backend == 'sequential':
        return 'sequential'
    
    if parallel_backend == 'numba':
        if not _NUMBA_AVAILABLE:
            raise ImportError(
                "Numba is required for parallel_backend='numba' but is not installed. "
                "Install with: pip install numba"
            )
        return 'numba'
    
    # auto
    if _NUMBA_AVAILABLE:  # Phase 12.11 fix
        return 'numba'
    return 'sequential'


def _prepare_chunk_matrices_for_numba(
    chunk_data,
    fit_cols,
    linear_cols_list,
    weights_list,
    n_params_list,
):
    """
    Convert chunk dict to dense matrices for Numba kernel.
    
    Parameters
    ----------
    chunk_data : dict[str, ndarray]
        Column data for this chunk (already sorted)
    fit_cols : list[str]
        Target column names
    linear_cols_list : list[list[str]]
        Linear columns per fit
    weights_list : list[str or None]
        Weight column per fit
    n_params_list : list[int]
        Number of parameters per fit
        
    Returns
    -------
    X_mat : ndarray[n_rows, n_linear_unique]
    Y_mat : ndarray[n_rows, n_targets_unique]
    W_mat : ndarray[n_rows, n_weights_unique]
    fit_target_idx : ndarray[n_fits]
    fit_weight_idx : ndarray[n_fits] (-1 means no weight)
    fit_n_params : ndarray[n_fits]
    linear_col_indices : ndarray[total_refs]
    linear_col_starts : ndarray[n_fits + 1]
    """
    n_fits = len(fit_cols)
    
    # Collect unique columns
    linear_cols_unique = sorted(set(c for lc in linear_cols_list for c in lc))
    target_cols_unique = sorted(set(fit_cols))
    weight_cols_unique = sorted(set(w for w in weights_list if w is not None))
    
    # Get chunk size
    first_col = next(iter(chunk_data.values()))
    n_rows = len(first_col)
    
    # Build dense matrices
    n_linear = len(linear_cols_unique)
    n_targets = len(target_cols_unique)
    n_weights = len(weight_cols_unique)
    
    X_mat = np.empty((n_rows, n_linear), dtype=np.float64)
    Y_mat = np.empty((n_rows, n_targets), dtype=np.float64)
    W_mat = np.empty((n_rows, max(n_weights, 1)), dtype=np.float64)
    
    # Fill matrices
    linear_idx = {}
    for j, col in enumerate(linear_cols_unique):
        X_mat[:, j] = chunk_data[col].astype(np.float64)
        linear_idx[col] = j
    
    target_idx = {}
    for j, col in enumerate(target_cols_unique):
        Y_mat[:, j] = chunk_data[col].astype(np.float64)
        target_idx[col] = j
    
    weight_idx = {}
    if weight_cols_unique:
        for j, col in enumerate(weight_cols_unique):
            W_mat[:, j] = chunk_data[col].astype(np.float64)
            weight_idx[col] = j
    else:
        W_mat[:, 0] = 1.0  # Default weights
    
    # Build fit specifications
    fit_target_idx = np.array([target_idx[fit_cols[fi]] for fi in range(n_fits)], dtype=np.int32)
    fit_weight_idx = np.array(
        [weight_idx.get(weights_list[fi], -1) if weights_list[fi] else -1 for fi in range(n_fits)],
        dtype=np.int32
    )
    fit_n_params = np.array(n_params_list, dtype=np.int32)
    
    # Build linear column indices (offset-based encoding)
    linear_col_indices = []
    linear_col_starts = [0]
    for fi in range(n_fits):
        for col in linear_cols_list[fi]:
            linear_col_indices.append(linear_idx[col])
        linear_col_starts.append(len(linear_col_indices))
    
    linear_col_indices = np.array(linear_col_indices, dtype=np.int32)
    linear_col_starts = np.array(linear_col_starts, dtype=np.int32)
    
    return (
        X_mat, Y_mat, W_mat,
        fit_target_idx, fit_weight_idx, fit_n_params,
        linear_col_indices, linear_col_starts,
    )


# Numba helper: median calculation
@njit(cache=True)
def _median_numba(arr):
    """Compute median of 1D array."""
    n = len(arr)
    if n == 0:
        return np.nan
    sorted_arr = np.sort(arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2.0


# Numba helper: MAD calculation
@njit(cache=True)
def _mad_numba(arr):
    """Compute Median Absolute Deviation of 1D array."""
    n = len(arr)
    if n == 0:
        return np.nan
    med = _median_numba(arr)
    abs_dev = np.abs(arr - med)
    return _median_numba(abs_dev)


# Numba helper: Cholesky solve
@njit(cache=True)
def _cholesky_solve_numba(L, b):
    """Solve L @ L.T @ x = b given Cholesky factor L."""
    n = len(b)
    # Forward solve: L @ y = b
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = b[i]
        for j in range(i):
            s -= L[i, j] * y[j]
        y[i] = s / L[i, i]
    # Backward solve: L.T @ x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        s = y[i]
        for j in range(i + 1, n):
            s -= L[j, i] * x[j]
        x[i] = s / L[i, i]
    return x


# Numba helper: Cholesky decomposition
@njit(cache=True)
def _cholesky_numba(A):
    """
    Compute Cholesky decomposition of positive definite matrix A.
    Returns (L, success) where L is lower triangular and success is bool.
    """
    n = A.shape[0]
    L = np.zeros((n, n), dtype=np.float64)
    
    for i in range(n):
        for j in range(i + 1):
            s = A[i, j]
            for k in range(j):
                s -= L[i, k] * L[j, k]
            if i == j:
                if s <= 0:
                    return L, False
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]
    
    return L, True


# Numba helper: process one fit for one group
@njit(cache=True)
def _process_one_fit_numba(
    gi_global, fi, i0, i1, m,
    X_mat, Y_mat, W_mat,
    target_idx, weight_idx, n_params,
    linear_col_indices, lin_start, lin_end,
    fit_intercept,
    beta, errors, rms_arr, mad_arr,
    n_valid_arr, n_filtered_arr, cond_arr, status_arr,
    min_stat, compute_mad,
):
    """Process one fit for one group using streaming accumulation."""
    n_lin = lin_end - lin_start
    
    # Initialize accumulators
    XtX = np.zeros((n_params, n_params), dtype=np.float64)
    XtY = np.zeros(n_params, dtype=np.float64)
    n_valid = 0
    
    # First pass: accumulate XtX, XtY with validity checking
    for row in range(i0, i1):
        # Get target value
        y_val = Y_mat[row, target_idx]
        if not np.isfinite(y_val):
            continue
        
        # Get weight
        if weight_idx >= 0:
            w_val = W_mat[row, weight_idx]
        else:
            w_val = 1.0
        if not (np.isfinite(w_val) and w_val > 0):
            continue
        
        # Check predictors are finite
        valid_row = True
        for k in range(n_lin):
            col_idx = linear_col_indices[lin_start + k]
            if not np.isfinite(X_mat[row, col_idx]):
                valid_row = False
                break
        if not valid_row:
            continue
        
        # This row is valid
        n_valid += 1
        sqrt_w = np.sqrt(w_val)
        
        # Build weighted x vector for this row
        if fit_intercept:
            x_w = np.empty(n_params, dtype=np.float64)
            x_w[0] = sqrt_w
            for k in range(n_lin):
                col_idx = linear_col_indices[lin_start + k]
                x_w[k + 1] = X_mat[row, col_idx] * sqrt_w
        else:
            x_w = np.empty(n_params, dtype=np.float64)
            for k in range(n_lin):
                col_idx = linear_col_indices[lin_start + k]
                x_w[k] = X_mat[row, col_idx] * sqrt_w
        
        y_w = y_val * sqrt_w
        
        # Accumulate XtX and XtY
        for p in range(n_params):
            XtY[p] += x_w[p] * y_w
            for q in range(p + 1):
                XtX[p, q] += x_w[p] * x_w[q]
    
    # Fill upper triangle of XtX
    for p in range(n_params):
        for q in range(p + 1, n_params):
            XtX[p, q] = XtX[q, p]
    
    # Store diagnostics
    n_filtered_arr[gi_global, fi] = m - n_valid
    n_valid_arr[gi_global, fi] = n_valid
    
    if n_valid < min_stat:
        status_arr[gi_global, fi] = STATUS_INSUFFICIENT_VALID
        return
    
    # Cholesky decomposition
    L, success = _cholesky_numba(XtX)
    
    if not success:
        status_arr[gi_global, fi] = STATUS_SINGULAR
        return
    
    # Condition proxy from Cholesky diagonal
    diag_min = L[0, 0]
    diag_max = L[0, 0]
    for p in range(1, n_params):
        if L[p, p] < diag_min:
            diag_min = L[p, p]
        if L[p, p] > diag_max:
            diag_max = L[p, p]
    
    cond_proxy = (diag_max / diag_min) ** 2 if diag_min > 0 else 1e30
    cond_arr[gi_global, fi] = cond_proxy
    
    # Check ill-conditioning
    if cond_proxy > 1e12:
        # Add ridge and re-factor
        ridge = 1e-8 * np.trace(XtX) / n_params
        for p in range(n_params):
            XtX[p, p] += ridge
        L, success = _cholesky_numba(XtX)
        if not success:
            status_arr[gi_global, fi] = STATUS_SINGULAR
            return
        status_arr[gi_global, fi] = STATUS_ILL_CONDITIONED
    else:
        status_arr[gi_global, fi] = STATUS_OK
    
    # Solve for coefficients
    coeffs = _cholesky_solve_numba(L, XtY)
    for p in range(n_params):
        beta[gi_global, fi, p] = coeffs[p]
    
    # Second pass: compute residuals for RMS and MAD
    rss = 0.0
    resid_uw = np.empty(n_valid, dtype=np.float64)
    resid_idx = 0
    
    for row in range(i0, i1):
        y_val = Y_mat[row, target_idx]
        if not np.isfinite(y_val):
            continue
        
        if weight_idx >= 0:
            w_val = W_mat[row, weight_idx]
        else:
            w_val = 1.0
        if not (np.isfinite(w_val) and w_val > 0):
            continue
        
        valid_row = True
        for k in range(n_lin):
            col_idx = linear_col_indices[lin_start + k]
            if not np.isfinite(X_mat[row, col_idx]):
                valid_row = False
                break
        if not valid_row:
            continue
        
        # Compute prediction
        y_pred = 0.0
        if fit_intercept:
            y_pred = coeffs[0]
            for k in range(n_lin):
                col_idx = linear_col_indices[lin_start + k]
                y_pred += coeffs[k + 1] * X_mat[row, col_idx]
        else:
            for k in range(n_lin):
                col_idx = linear_col_indices[lin_start + k]
                y_pred += coeffs[k] * X_mat[row, col_idx]
        
        # Unweighted residual (for MAD)
        resid = y_val - y_pred
        resid_uw[resid_idx] = resid
        resid_idx += 1
        
        # Weighted residual for RSS
        sqrt_w = np.sqrt(w_val)
        rss += (resid * sqrt_w) ** 2
    
    # Compute RMS
    dof = n_valid - n_params
    if dof > 0:
        s2 = rss / dof
        rms_arr[gi_global, fi] = np.sqrt(s2)
        
        # Compute parameter errors from XtX_inv
        # L @ L.T = XtX, so XtX_inv = L.T_inv @ L_inv
        # Compute diagonal of XtX_inv
        L_inv = np.zeros((n_params, n_params), dtype=np.float64)
        for i in range(n_params):
            L_inv[i, i] = 1.0 / L[i, i]
            for j in range(i + 1, n_params):
                s = 0.0
                for k in range(i, j):
                    s += L[j, k] * L_inv[k, i]
                L_inv[j, i] = -s / L[j, j]
        
        # Diagonal of XtX_inv = sum of squared columns of L_inv
        for p in range(n_params):
            var_p = 0.0
            for k in range(p, n_params):
                var_p += L_inv[k, p] ** 2
            errors[gi_global, fi, p] = np.sqrt(s2 * var_p)
    
    # Compute MAD
    if compute_mad:
        mad_arr[gi_global, fi] = _mad_numba(resid_uw)


# Main Numba kernel: parallel over groups
@njit(parallel=True, cache=True)
def _process_chunk_numba(
    X_mat, Y_mat, W_mat,
    local_offsets,
    fit_target_idx, fit_weight_idx, fit_n_params,
    linear_col_indices, linear_col_starts,
    fit_intercept,
    beta, errors, rms_arr, mad_arr,
    n_total_arr, n_valid_arr, n_filtered_arr, cond_arr, status_arr,
    gi_offset,
    min_stat, compute_mad,
):
    """
    Process all groups in a chunk using Numba parallel.
    
    Phase 12.9.GB: Uses prange for parallel group processing.
    """
    n_groups_chunk = len(local_offsets) - 1
    n_fits = len(fit_target_idx)
    
    for gi_local in prange(n_groups_chunk):
        gi_global = gi_offset + gi_local
        i0 = local_offsets[gi_local]
        i1 = local_offsets[gi_local + 1]
        m = i1 - i0
        
        n_total_arr[gi_global] = m
        
        if m < min_stat:
            for fi in range(n_fits):
                status_arr[gi_global, fi] = STATUS_INSUFFICIENT_DATA
            continue
        
        # Process each fit for this group
        for fi in range(n_fits):
            _process_one_fit_numba(
                gi_global, fi, i0, i1, m,
                X_mat, Y_mat, W_mat,
                fit_target_idx[fi], fit_weight_idx[fi], fit_n_params[fi],
                linear_col_indices, linear_col_starts[fi], linear_col_starts[fi + 1],
                fit_intercept,
                beta, errors, rms_arr, mad_arr,
                n_valid_arr, n_filtered_arr, cond_arr, status_arr,
                min_stat, compute_mad,
            )


def _build_empty_v5_result(
    gb_cols, fit_cols, suffixes_list, linear_cols_list,
    fit_intercept, compute_mad, diag, diag_prefix, return_metadata
):
    """Build empty result DataFrame and metadata for empty input."""
    out_dict = {col: [] for col in gb_cols}
    
    n_fits = len(fit_cols)
    for fi in range(n_fits):
        target = fit_cols[fi]
        suffix = suffixes_list[fi]
        lin_cols = linear_cols_list[fi]
        
        if fit_intercept:
            out_dict[f"{target}_intercept{suffix}"] = []
            out_dict[f"{target}_intercept_err{suffix}"] = []
        for lc in lin_cols:
            out_dict[f"{target}_slope_{lc}{suffix}"] = []
            out_dict[f"{target}_slope_{lc}_err{suffix}"] = []
        out_dict[f"{target}_rms{suffix}"] = []
        out_dict[f"{target}_mad{suffix}"] = []
    
    if diag:
        # Shared: n_total only
        out_dict[f"{diag_prefix}n_total_v5"] = []
        # Per-fit: n_valid, n_filtered, cond, status (Phase 12.9.GB)
        for fi in range(n_fits):
            suffix = suffixes_list[fi]
            out_dict[f"{diag_prefix}n_valid{suffix}"] = []
            out_dict[f"{diag_prefix}n_filtered{suffix}"] = []
            out_dict[f"{diag_prefix}cond{suffix}"] = []
            out_dict[f"{diag_prefix}status{suffix}"] = []
    
    dfGB = pd.DataFrame(out_dict)
    
    if return_metadata:
        metadata = {
            "version": "v5",
            "schema_version": "1.0",
            "n_fits": n_fits,
            "n_groups": 0,
            "fits": {},
        }
        return dfGB, metadata
    
    return dfGB


def _build_v5_metadata(
    gb_cols,
    fit_cols,
    suffixes_list,
    linear_cols_list,
    weights_list,
    fit_intercept,
    min_stat,
    compute_mad,
    n_groups,
):
    """Build metadata dict with per-fit formulas."""
    metadata = {
        "version": "v5",
        "schema_version": "1.0",
        "gb_columns": gb_cols,
        "n_fits": len(fit_cols),
        "n_groups": n_groups,
        "fit_intercept": fit_intercept,
        "min_stat": min_stat,
        "compute_mad": compute_mad,
        "fits": {},
    }
    
    for fi, (target, suffix, lin_cols, weight) in enumerate(
        zip(fit_cols, suffixes_list, linear_cols_list, weights_list)
    ):
        # Build column names
        columns = {
            "rms": f"{target}_rms{suffix}",
            "mad": f"{target}_mad{suffix}",
        }
        
        slopes = []
        slope_errs = []
        
        if fit_intercept:
            columns["intercept"] = f"{target}_intercept{suffix}"
            columns["intercept_err"] = f"{target}_intercept_err{suffix}"
        
        for lc in lin_cols:
            slopes.append(f"{target}_slope_{lc}{suffix}")
            slope_errs.append(f"{target}_slope_{lc}_err{suffix}")
        
        columns["slopes"] = slopes
        columns["slope_errs"] = slope_errs
        
        # Build prediction formula
        terms = []
        if fit_intercept:
            terms.append(f"{target}_intercept{suffix}")
        for lc in lin_cols:
            terms.append(f"{target}_slope_{lc}{suffix} * {lc}")
        prediction = " + ".join(terms)
        
        # Build residual formula
        residual = f"{target} - ({prediction})"
        
        # Build pull formula
        pull = f"({residual}) / {target}_rms{suffix}"
        
        metadata["fits"][suffix] = {
            "target": target,
            "linear_columns": lin_cols,
            "weights": weight,
            "n_params": (1 + len(lin_cols)) if fit_intercept else len(lin_cols),
            "columns": columns,
            "formulas": {
                "prediction": prediction,
                "residual": residual,
                "pull": pull,
            },
        }
    
    return metadata
