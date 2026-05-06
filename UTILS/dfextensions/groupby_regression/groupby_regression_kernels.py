"""
Phase 12.14.GB: Shared Numba Kernels for GroupBy Regression

This module provides optimized Numba JIT kernels for V4/V5/V6 groupby 
regression functions. It implements:

1. Single-fit kernel: Baseline, always works, supports filtering
2. Multi-fit kernel: Optimized, shares XtX across targets with same linear_columns
3. Dispatcher: Selects appropriate kernel based on context
4. Status bitmask utilities: Encode/decode diagnostic status

KERNEL INTERFACE FROZEN FOR PHASE 12.x
Any signature change requires architect review and benchmark re-approval.

Author: Team 3 Coder
Date: 2025-12-31
Phase: 12.14.GB
"""

import numpy as np
import warnings
from typing import Tuple, Optional

# ============================================================================
# NUMBA DETECTION
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
# STATUS BITMASK CONSTANTS (per PHASE_12_14_STATUS_BITMASK_SPEC_v2.1)
# ============================================================================

# Status bit constants (Numba-compatible uint8)
STATUS_OK               = np.uint8(0)
STATUS_INSUFFICIENT     = np.uint8(1 << 0)   # 1: m_valid < min_stat
STATUS_UNDERDETERMINED  = np.uint8(1 << 1)   # 2: m_valid <= n_params
STATUS_XW_INVALID       = np.uint8(1 << 2)   # 4: NaN/Inf in X or W
STATUS_Y_INVALID        = np.uint8(1 << 3)   # 8: NaN/Inf in Y
STATUS_SINGULAR         = np.uint8(1 << 4)   # 16: XtX ill-conditioned, stabilized
STATUS_NUMERICAL_ERROR  = np.uint8(1 << 5)   # 32: NaN/Inf in result
STATUS_DIAG_INVALID     = np.uint8(1 << 6)   # 64: MAD/RMS computation failed

# Blocking mask (any of these → beta/errors = NaN)
STATUS_BLOCKING = (
    STATUS_INSUFFICIENT | 
    STATUS_UNDERDETERMINED | 
    STATUS_NUMERICAL_ERROR
)

# Data validity mask
STATUS_DATA_INVALID = STATUS_XW_INVALID | STATUS_Y_INVALID

# Invalid handling modes (Numba-compatible integers)
INVALID_ASSUME_CLEAN = 0  # No checks (fastest)
INVALID_DETECT = 1        # Check + set bits, no filtering (DEFAULT)
INVALID_FILTER = 2        # Check + set bits + skip invalid rows

# Condition number threshold for ridge stabilization
COND_THRESHOLD = 1e10

# Status name mapping for decode_status
_STATUS_NAMES = [
    (STATUS_INSUFFICIENT, 'INSUFFICIENT'),
    (STATUS_UNDERDETERMINED, 'UNDERDETERMINED'),
    (STATUS_XW_INVALID, 'XW_INVALID'),
    (STATUS_Y_INVALID, 'Y_INVALID'),
    (STATUS_SINGULAR, 'SINGULAR'),
    (STATUS_NUMERICAL_ERROR, 'NUMERICAL_ERROR'),
    (STATUS_DIAG_INVALID, 'DIAG_INVALID'),
]


# ============================================================================
# STATUS DECODING (Vectorized for performance)
# ============================================================================

def decode_status(mask):
    """
    Convert bitmask to human-readable string (vectorized).
    
    Parameters
    ----------
    mask : int or array-like
        Status bitmask value(s)
    
    Returns
    -------
    str or ndarray
        'OK' if mask==0, else '|'-separated issue names
    
    Examples
    --------
    >>> decode_status(0)
    'OK'
    >>> decode_status(17)  # INSUFFICIENT | SINGULAR
    'INSUFFICIENT|SINGULAR'
    >>> decode_status(np.array([0, 1, 16]))
    array(['OK', 'INSUFFICIENT', 'SINGULAR'], dtype='<U32')
    """
    if np.isscalar(mask):
        return _decode_single(int(mask))
    return np.vectorize(_decode_single, otypes=[str])(mask)


def _decode_single(mask: int) -> str:
    """Decode a single status mask value."""
    if mask == 0:
        return 'OK'
    
    issues = []
    for bit, name in _STATUS_NAMES:
        if mask & bit:
            issues.append(name)
    
    return '|'.join(issues) if issues else 'UNKNOWN'


# ============================================================================
# NUMBA HELPER FUNCTIONS
# ============================================================================

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


@njit(cache=True)
def _mad_numba(arr):
    """Compute Median Absolute Deviation of 1D array."""
    n = len(arr)
    if n == 0:
        return np.nan
    med = _median_numba(arr)
    abs_dev = np.abs(arr - med)
    return _median_numba(abs_dev)


@njit(cache=True)
def _cholesky_numba(A):
    """
    Compute Cholesky decomposition of positive definite matrix A.
    
    Returns
    -------
    L : ndarray
        Lower triangular Cholesky factor
    success : bool
        True if decomposition succeeded
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


# ============================================================================
# SINGLE-FIT KERNEL (Baseline, always works)
# ============================================================================

@njit(cache=True)
def _fit_one_group_single(
    X_slice, Y_slice, W_slice,
    n_feat, n_params, fit_intercept,
    min_stat, compute_mad, invalid_handling,
):
    """
    Fit a single target for a single group.
    
    This is the baseline kernel that always works, including with filtering.
    
    Parameters
    ----------
    X_slice : ndarray (m, n_feat)
        Predictor values for this group
    Y_slice : ndarray (m,)
        Target values for this group
    W_slice : ndarray (m,) or None
        Weight values (None for unweighted)
    n_feat : int
        Number of predictor columns
    n_params : int
        Number of parameters (n_feat + intercept if fit_intercept)
    fit_intercept : bool
        Whether to fit intercept
    min_stat : int
        Minimum valid samples required
    compute_mad : bool
        Whether to compute MAD
    invalid_handling : int
        INVALID_ASSUME_CLEAN, INVALID_DETECT, or INVALID_FILTER
    
    Returns
    -------
    beta : ndarray (n_params,)
        Fitted coefficients
    errors : ndarray (n_params,)
        Standard errors
    rms : float
        Root mean square error
    mad : float
        Median absolute deviation (NaN if not computed)
    status : uint8
        Status bitmask
    n_valid : int
        Number of valid samples used
    n_filtered : int
        Number of samples filtered out
    cond : float
        Condition number proxy
    """
    m = len(Y_slice)
    has_weights = W_slice is not None
    
    # Output arrays
    beta = np.full(n_params, np.nan, dtype=np.float64)
    errors = np.full(n_params, np.nan, dtype=np.float64)
    rms = np.nan
    mad = np.nan
    status = STATUS_OK
    cond = np.nan
    
    # Detection pass (if not assume_clean)
    has_xw_invalid = False
    has_y_invalid = False
    
    if invalid_handling != INVALID_ASSUME_CLEAN:
        for row in range(m):
            # Check Y
            if not np.isfinite(Y_slice[row]):
                has_y_invalid = True
            # Check X
            for k in range(n_feat):
                if not np.isfinite(X_slice[row, k]):
                    has_xw_invalid = True
                    break
            # Check W
            if has_weights and not np.isfinite(W_slice[row]):
                has_xw_invalid = True
        
        if has_xw_invalid:
            status |= STATUS_XW_INVALID
        if has_y_invalid:
            status |= STATUS_Y_INVALID
    
    # Accumulate XtX, XtY with optional filtering
    # Phase 13.21.GB-PERF: also accumulate sum_y, sum_y2 for vectorized R²
    XtX = np.zeros((n_params, n_params), dtype=np.float64)
    XtY = np.zeros(n_params, dtype=np.float64)
    n_valid = 0
    sum_y = 0.0
    sum_y2 = 0.0
    
    for row in range(m):
        y_val = Y_slice[row]
        
        # Check validity (always if filtering, otherwise only if not assume_clean)
        if invalid_handling == INVALID_FILTER:
            if not np.isfinite(y_val):
                continue
            
            valid_row = True
            for k in range(n_feat):
                if not np.isfinite(X_slice[row, k]):
                    valid_row = False
                    break
            if not valid_row:
                continue
            
            if has_weights:
                w_val = W_slice[row]
                if not (np.isfinite(w_val) and w_val > 0):
                    continue
            else:
                w_val = 1.0
        else:
            # No filtering - use all rows
            if has_weights:
                w_val = W_slice[row]
            else:
                w_val = 1.0
        
        n_valid += 1
        sum_y += y_val
        sum_y2 += y_val * y_val
        sqrt_w = np.sqrt(w_val) if w_val > 0 else 0.0
        
        # Build weighted x vector
        if fit_intercept:
            x_w = np.empty(n_params, dtype=np.float64)
            x_w[0] = sqrt_w
            for k in range(n_feat):
                x_w[k + 1] = X_slice[row, k] * sqrt_w
        else:
            x_w = np.empty(n_params, dtype=np.float64)
            for k in range(n_feat):
                x_w[k] = X_slice[row, k] * sqrt_w
        
        y_w = y_val * sqrt_w
        
        # Accumulate XtX (lower triangle) and XtY
        for p in range(n_params):
            XtY[p] += x_w[p] * y_w
            for q in range(p + 1):
                XtX[p, q] += x_w[p] * x_w[q]
    
    # Fill upper triangle
    for p in range(n_params):
        for q in range(p + 1, n_params):
            XtX[p, q] = XtX[q, p]
    
    n_filtered = m - n_valid
    
    # Check sufficient data
    if n_valid < min_stat:
        status |= STATUS_INSUFFICIENT
        return beta, errors, rms, mad, status, n_valid, n_filtered, cond, sum_y, sum_y2
    
    if n_valid <= n_params:
        status |= STATUS_UNDERDETERMINED
        return beta, errors, rms, mad, status, n_valid, n_filtered, cond, sum_y, sum_y2
    
    # Cholesky decomposition
    L, success = _cholesky_numba(XtX)
    
    if not success:
        status |= STATUS_NUMERICAL_ERROR
        return beta, errors, rms, mad, status, n_valid, n_filtered, cond, sum_y, sum_y2
    
    # Condition proxy from Cholesky diagonal
    diag_min = L[0, 0]
    diag_max = L[0, 0]
    for p in range(1, n_params):
        if L[p, p] < diag_min:
            diag_min = L[p, p]
        if L[p, p] > diag_max:
            diag_max = L[p, p]
    
    cond_proxy = (diag_max / diag_min) ** 2 if diag_min > 0 else 1e30
    cond = cond_proxy
    
    # Check ill-conditioning
    if cond_proxy > COND_THRESHOLD:
        # Apply ridge stabilization
        ridge = 1e-8 * np.trace(XtX) / n_params
        for p in range(n_params):
            XtX[p, p] += ridge
        L, success = _cholesky_numba(XtX)
        if not success:
            status |= STATUS_NUMERICAL_ERROR
            return beta, errors, rms, mad, status, n_valid, n_filtered, cond, sum_y, sum_y2
        status |= STATUS_SINGULAR
    
    # Solve for coefficients
    coeffs = _cholesky_solve_numba(L, XtY)
    for p in range(n_params):
        beta[p] = coeffs[p]
    
    # Check for NaN in result
    for p in range(n_params):
        if not np.isfinite(beta[p]):
            status |= STATUS_NUMERICAL_ERROR
            return beta, errors, rms, mad, status, n_valid, n_filtered, cond, sum_y, sum_y2
    
    # Second pass: compute residuals for RMS and MAD
    rss = 0.0
    resid_uw = np.empty(n_valid, dtype=np.float64)
    resid_idx = 0
    
    for row in range(m):
        y_val = Y_slice[row]
        
        # Same filtering logic
        if invalid_handling == INVALID_FILTER:
            if not np.isfinite(y_val):
                continue
            valid_row = True
            for k in range(n_feat):
                if not np.isfinite(X_slice[row, k]):
                    valid_row = False
                    break
            if not valid_row:
                continue
            if has_weights:
                w_val = W_slice[row]
                if not (np.isfinite(w_val) and w_val > 0):
                    continue
            else:
                w_val = 1.0
        else:
            if has_weights:
                w_val = W_slice[row]
            else:
                w_val = 1.0
        
        # Compute prediction
        y_pred = 0.0
        if fit_intercept:
            y_pred = coeffs[0]
            for k in range(n_feat):
                y_pred += coeffs[k + 1] * X_slice[row, k]
        else:
            for k in range(n_feat):
                y_pred += coeffs[k] * X_slice[row, k]
        
        # Residual
        resid = y_val - y_pred
        resid_uw[resid_idx] = resid
        resid_idx += 1
        
        # Weighted residual for RSS
        sqrt_w = np.sqrt(w_val) if w_val > 0 else 0.0
        rss += (resid * sqrt_w) ** 2
    
    # Compute RMS and errors
    dof = n_valid - n_params
    if dof > 0:
        s2 = rss / dof
        rms = np.sqrt(s2)
        
        # Compute parameter errors from L_inv
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
            errors[p] = np.sqrt(s2 * var_p)
    
    # Compute MAD
    if compute_mad:
        if resid_idx > 0:
            mad = _mad_numba(resid_uw[:resid_idx])
        else:
            status |= STATUS_DIAG_INVALID
    
    return beta, errors, rms, mad, status, n_valid, n_filtered, cond, sum_y, sum_y2


@njit(parallel=True, cache=True)
def fit_groups_single_numba(
    X_all, Y_all, W_all, offsets,
    n_groups, n_feat, n_params,
    fit_intercept, min_stat, compute_mad, invalid_handling,
    out_beta, out_errors, out_rms, out_mad,
    out_status, out_n_valid, out_n_filtered, out_cond,
    out_sum_y, out_sum_y2,
):
    """
    Process all groups with single-fit kernel (parallel over groups).
    
    Phase 13.21.GB-PERF: extended with out_sum_y / out_sum_y2 outputs
    for vectorized R² computation (eliminates per-bin np.mean/np.sum
    in the Python wrapper).
    
    Parameters
    ----------
    X_all : ndarray (n_rows, n_feat)
        All predictor values, sorted by group
    Y_all : ndarray (n_rows,)
        All target values, sorted by group
    W_all : ndarray (n_rows,) or empty
        All weight values (empty for unweighted)
    offsets : ndarray (n_groups + 1,)
        Group boundary offsets
    n_groups : int
        Number of groups
    n_feat : int
        Number of predictor columns
    n_params : int
        Number of parameters
    fit_intercept : bool
        Whether to fit intercept
    min_stat : int
        Minimum valid samples required
    compute_mad : bool
        Whether to compute MAD
    invalid_handling : int
        INVALID_ASSUME_CLEAN, INVALID_DETECT, or INVALID_FILTER
    out_* : ndarray
        Pre-allocated output arrays
    out_sum_y : ndarray (n_groups,) float64
        Per-group sum of valid Y values (for R²)
    out_sum_y2 : ndarray (n_groups,) float64
        Per-group sum of valid Y² values (for R²)
    """
    has_weights = len(W_all) > 0
    
    for gi in prange(n_groups):
        i0 = offsets[gi]
        i1 = offsets[gi + 1]
        
        X_slice = X_all[i0:i1]
        Y_slice = Y_all[i0:i1]
        
        if has_weights:
            W_slice = W_all[i0:i1]
        else:
            W_slice = None
        
        beta, errors, rms, mad, status, n_valid, n_filtered, cond, sum_y, sum_y2 = _fit_one_group_single(
            X_slice, Y_slice, W_slice,
            n_feat, n_params, fit_intercept,
            min_stat, compute_mad, invalid_handling,
        )
        
        for p in range(n_params):
            out_beta[gi, p] = beta[p]
            out_errors[gi, p] = errors[p]
        out_rms[gi] = rms
        out_mad[gi] = mad
        out_status[gi] = status
        out_n_valid[gi] = n_valid
        out_n_filtered[gi] = n_filtered
        out_cond[gi] = cond
        out_sum_y[gi] = sum_y
        out_sum_y2[gi] = sum_y2


# ============================================================================
# MULTI-FIT KERNEL (Optimized, shares XtX across targets)
# ============================================================================

@njit(cache=True)
def _fit_one_group_multifit(
    X_slice, Y_slice, W_slice,
    n_feat, n_tgt, n_params, fit_intercept,
    min_stat, compute_mad, invalid_handling,
):
    """
    Fit multiple targets for a single group, sharing XtX computation.
    
    This kernel is optimized for the common case where all targets share
    the same linear_columns and weights. XtX is computed once and reused.
    
    IMPORTANT: This kernel does NOT support invalid_handling='filter' because
    different Y columns may have different NaN patterns, breaking XtX sharing.
    
    Parameters
    ----------
    X_slice : ndarray (m, n_feat)
        Predictor values for this group
    Y_slice : ndarray (m, n_tgt)
        Target values for this group (multiple columns)
    W_slice : ndarray (m,) or None
        Weight values
    n_feat : int
        Number of predictor columns
    n_tgt : int
        Number of target columns
    n_params : int
        Number of parameters
    fit_intercept : bool
        Whether to fit intercept
    min_stat : int
        Minimum valid samples required
    compute_mad : bool
        Whether to compute MAD
    invalid_handling : int
        INVALID_ASSUME_CLEAN or INVALID_DETECT (NOT FILTER)
    
    Returns
    -------
    beta : ndarray (n_tgt, n_params)
    errors : ndarray (n_tgt, n_params)
    rms : ndarray (n_tgt,)
    mad : ndarray (n_tgt,)
    status : ndarray (n_tgt,) uint8
    n_valid : int (shared across targets since no filtering)
    n_filtered : int
    cond : float
    """
    m = len(X_slice)
    has_weights = W_slice is not None
    
    # Output arrays
    beta = np.full((n_tgt, n_params), np.nan, dtype=np.float64)
    errors = np.full((n_tgt, n_params), np.nan, dtype=np.float64)
    rms = np.full(n_tgt, np.nan, dtype=np.float64)
    mad = np.full(n_tgt, np.nan, dtype=np.float64)
    status = np.zeros(n_tgt, dtype=np.uint8)
    cond = np.nan
    
    # Detection pass for X and W (shared across targets)
    has_xw_invalid = False
    
    if invalid_handling != INVALID_ASSUME_CLEAN:
        for row in range(m):
            for k in range(n_feat):
                if not np.isfinite(X_slice[row, k]):
                    has_xw_invalid = True
                    break
            if has_weights and not np.isfinite(W_slice[row]):
                has_xw_invalid = True
        
        if has_xw_invalid:
            for t in range(n_tgt):
                status[t] |= STATUS_XW_INVALID
    
    # Detection pass for Y (per-target)
    if invalid_handling != INVALID_ASSUME_CLEAN:
        for t in range(n_tgt):
            has_y_invalid = False
            for row in range(m):
                if not np.isfinite(Y_slice[row, t]):
                    has_y_invalid = True
                    break
            if has_y_invalid:
                status[t] |= STATUS_Y_INVALID
    
    # Build XtX once (shared across all targets) and XtY for all targets simultaneously
    # NO x_weighted storage - recompute weighted x on-the-fly to avoid memory allocation
    XtX = np.zeros((n_params, n_params), dtype=np.float64)
    XtY_all = np.zeros((n_tgt, n_params), dtype=np.float64)  # All targets at once
    n_valid = 0
    
    for row in range(m):
        if has_weights:
            w_val = W_slice[row]
        else:
            w_val = 1.0
        
        if w_val <= 0:
            continue
        
        n_valid += 1
        sqrt_w = np.sqrt(w_val)
        
        # Build weighted x vector on-the-fly (no storage)
        # Accumulate XtX and XtY for all targets in one pass
        if fit_intercept:
            # XtX accumulation
            XtX[0, 0] += sqrt_w * sqrt_w
            for k in range(n_feat):
                x_wk = X_slice[row, k] * sqrt_w
                XtX[k + 1, 0] += x_wk * sqrt_w
                for j in range(k + 1):
                    XtX[k + 1, j + 1] += x_wk * X_slice[row, j] * sqrt_w
            
            # XtY for all targets
            for t in range(n_tgt):
                y_w = Y_slice[row, t] * sqrt_w
                XtY_all[t, 0] += sqrt_w * y_w
                for k in range(n_feat):
                    XtY_all[t, k + 1] += X_slice[row, k] * sqrt_w * y_w
        else:
            # XtX accumulation (no intercept)
            for k in range(n_feat):
                x_wk = X_slice[row, k] * sqrt_w
                for j in range(k + 1):
                    XtX[k, j] += x_wk * X_slice[row, j] * sqrt_w
            
            # XtY for all targets
            for t in range(n_tgt):
                y_w = Y_slice[row, t] * sqrt_w
                for k in range(n_feat):
                    XtY_all[t, k] += X_slice[row, k] * sqrt_w * y_w
    
    # Fill upper triangle
    for p in range(n_params):
        for q in range(p + 1, n_params):
            XtX[p, q] = XtX[q, p]
    
    n_filtered = m - n_valid
    
    # Check sufficient data (shared check)
    if n_valid < min_stat:
        for t in range(n_tgt):
            status[t] |= STATUS_INSUFFICIENT
        return beta, errors, rms, mad, status, n_valid, n_filtered, cond
    
    if n_valid <= n_params:
        for t in range(n_tgt):
            status[t] |= STATUS_UNDERDETERMINED
        return beta, errors, rms, mad, status, n_valid, n_filtered, cond
    
    # Cholesky decomposition (shared)
    L, success = _cholesky_numba(XtX)
    
    if not success:
        for t in range(n_tgt):
            status[t] |= STATUS_NUMERICAL_ERROR
        return beta, errors, rms, mad, status, n_valid, n_filtered, cond
    
    # Condition proxy
    diag_min = L[0, 0]
    diag_max = L[0, 0]
    for p in range(1, n_params):
        if L[p, p] < diag_min:
            diag_min = L[p, p]
        if L[p, p] > diag_max:
            diag_max = L[p, p]
    
    cond_proxy = (diag_max / diag_min) ** 2 if diag_min > 0 else 1e30
    cond = cond_proxy
    
    # Ridge stabilization if needed (shared)
    if cond_proxy > COND_THRESHOLD:
        ridge = 1e-8 * np.trace(XtX) / n_params
        for p in range(n_params):
            XtX[p, p] += ridge
        L, success = _cholesky_numba(XtX)
        if not success:
            for t in range(n_tgt):
                status[t] |= STATUS_NUMERICAL_ERROR
            return beta, errors, rms, mad, status, n_valid, n_filtered, cond
        for t in range(n_tgt):
            status[t] |= STATUS_SINGULAR
    
    # Compute L_inv for errors (shared)
    L_inv = np.zeros((n_params, n_params), dtype=np.float64)
    for i in range(n_params):
        L_inv[i, i] = 1.0 / L[i, i]
        for j in range(i + 1, n_params):
            s = 0.0
            for k in range(i, j):
                s += L[j, k] * L_inv[k, i]
            L_inv[j, i] = -s / L[j, j]
    
    # Now solve for each target (XtY already computed)
    for t in range(n_tgt):
        # Get XtY for this target
        XtY = XtY_all[t]
        
        # Solve for coefficients
        coeffs = _cholesky_solve_numba(L, XtY)
        
        # Check for NaN
        has_nan = False
        for p in range(n_params):
            beta[t, p] = coeffs[p]
            if not np.isfinite(coeffs[p]):
                has_nan = True
        
        if has_nan:
            status[t] |= STATUS_NUMERICAL_ERROR
            continue
        
        # Compute residuals and RMS
        rss = 0.0
        resid_uw = np.empty(n_valid, dtype=np.float64)
        resid_idx = 0
        
        for row in range(m):
            if has_weights:
                w_val = W_slice[row]
            else:
                w_val = 1.0
            
            if w_val <= 0:
                continue
            
            y_val = Y_slice[row, t]
            
            # Compute prediction
            y_pred = 0.0
            if fit_intercept:
                y_pred = coeffs[0]
                for k in range(n_feat):
                    y_pred += coeffs[k + 1] * X_slice[row, k]
            else:
                for k in range(n_feat):
                    y_pred += coeffs[k] * X_slice[row, k]
            
            resid = y_val - y_pred
            resid_uw[resid_idx] = resid
            resid_idx += 1
            
            sqrt_w = np.sqrt(w_val)
            rss += (resid * sqrt_w) ** 2
        
        # RMS and errors
        dof = n_valid - n_params
        if dof > 0:
            s2 = rss / dof
            rms[t] = np.sqrt(s2)
            
            # Errors from L_inv
            for p in range(n_params):
                var_p = 0.0
                for k in range(p, n_params):
                    var_p += L_inv[k, p] ** 2
                errors[t, p] = np.sqrt(s2 * var_p)
        
        # MAD
        if compute_mad:
            if resid_idx > 0:
                mad[t] = _mad_numba(resid_uw[:resid_idx])
            else:
                status[t] |= STATUS_DIAG_INVALID
    
    return beta, errors, rms, mad, status, n_valid, n_filtered, cond


@njit(parallel=True, cache=True)
def fit_groups_multifit_numba(
    X_all, Y_all, W_all, offsets,
    n_groups, n_feat, n_tgt, n_params,
    fit_intercept, min_stat, compute_mad,
    out_beta, out_errors, out_rms, out_mad,
    out_status, out_n_valid, out_n_filtered, out_cond,
):
    """
    Process all groups with multi-fit kernel (parallel over groups).
    
    This kernel processes multiple targets simultaneously, sharing XtX
    computation across all targets. This provides 1.5-3× speedup for
    typical workloads with 3-6 targets.
    
    IMPORTANT: This kernel assumes invalid_handling is NOT 'filter'.
    Caller must validate this before invoking.
    
    Parameters
    ----------
    X_all : ndarray (n_rows, n_feat)
        All predictor values, sorted by group
    Y_all : ndarray (n_rows, n_tgt)
        All target values, sorted by group
    W_all : ndarray (n_rows,) or empty
        All weight values
    offsets : ndarray (n_groups + 1,)
        Group boundary offsets
    n_groups : int
        Number of groups
    n_feat : int
        Number of predictor columns
    n_tgt : int
        Number of target columns
    n_params : int
        Number of parameters
    fit_intercept : bool
        Whether to fit intercept
    min_stat : int
        Minimum valid samples required
    compute_mad : bool
        Whether to compute MAD
    out_* : ndarray
        Pre-allocated output arrays
    """
    has_weights = len(W_all) > 0
    
    # Note: invalid_handling is hardcoded to INVALID_DETECT for multi-fit
    # because filtering is not supported (different valid rows per target)
    invalid_handling = INVALID_DETECT
    
    for gi in prange(n_groups):
        i0 = offsets[gi]
        i1 = offsets[gi + 1]
        
        X_slice = X_all[i0:i1]
        Y_slice = Y_all[i0:i1]
        
        if has_weights:
            W_slice = W_all[i0:i1]
        else:
            W_slice = None
        
        beta, errors, rms, mad, status, n_valid, n_filtered, cond = _fit_one_group_multifit(
            X_slice, Y_slice, W_slice,
            n_feat, n_tgt, n_params, fit_intercept,
            min_stat, compute_mad, invalid_handling,
        )
        
        for t in range(n_tgt):
            for p in range(n_params):
                out_beta[gi, t, p] = beta[t, p]
                out_errors[gi, t, p] = errors[t, p]
            out_rms[gi, t] = rms[t]
            out_mad[gi, t] = mad[t]
            out_status[gi, t] = status[t]
        
        # Shared across targets
        out_n_valid[gi] = n_valid
        out_n_filtered[gi] = n_filtered
        out_cond[gi] = cond


# ============================================================================
# DISPATCHER (Selects appropriate kernel)
# ============================================================================

def fit_groups_dispatch(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    W_all: np.ndarray,
    offsets: np.ndarray,
    n_groups: int,
    n_feat: int,
    n_params: int,
    fit_intercept: bool,
    min_stat: int,
    compute_mad: bool,
    invalid_handling: int,
    n_targets: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Dispatch to appropriate kernel based on context.
    
    Decision logic:
    - n_targets == 1 → single-fit kernel
    - invalid_handling == INVALID_FILTER → single-fit kernel (per-target filtering)
    - Otherwise → multi-fit kernel (shares XtX)
    
    Parameters
    ----------
    X_all : ndarray (n_rows, n_feat)
        Predictor values
    Y_all : ndarray (n_rows,) or (n_rows, n_targets)
        Target values
    W_all : ndarray (n_rows,) or empty
        Weight values
    offsets : ndarray (n_groups + 1,)
        Group boundaries
    n_groups : int
        Number of groups
    n_feat : int
        Number of predictors
    n_params : int
        Number of parameters
    fit_intercept : bool
        Whether to fit intercept
    min_stat : int
        Minimum samples
    compute_mad : bool
        Whether to compute MAD
    invalid_handling : int
        Handling mode
    n_targets : int
        Number of target columns (default 1)
    
    Returns
    -------
    beta : ndarray
    errors : ndarray
    rms : ndarray
    mad : ndarray
    status : ndarray (uint8)
    n_valid : ndarray
    n_filtered : ndarray
    cond : ndarray
    """
    # Check Numba availability
    if not _NUMBA_AVAILABLE:
        warnings.warn(
            "Numba not available. Using NumPy fallback which is ~10× slower. "
            "Install Numba for optimal performance: pip install numba",
            RuntimeWarning
        )
        return _fit_groups_numpy_fallback(
            X_all, Y_all, W_all, offsets,
            n_groups, n_feat, n_params,
            fit_intercept, min_stat, compute_mad,
            invalid_handling, n_targets,
        )
    
    # Dispatch logic
    use_multifit = (n_targets > 1 and invalid_handling != INVALID_FILTER)
    
    if use_multifit:
        # Multi-fit kernel (shares XtX)
        # Ensure Y is 2D
        if Y_all.ndim == 1:
            Y_all = Y_all.reshape(-1, 1)
        
        # Allocate output arrays
        out_beta = np.empty((n_groups, n_targets, n_params), dtype=np.float64)
        out_errors = np.empty((n_groups, n_targets, n_params), dtype=np.float64)
        out_rms = np.empty((n_groups, n_targets), dtype=np.float64)
        out_mad = np.empty((n_groups, n_targets), dtype=np.float64)
        out_status = np.empty((n_groups, n_targets), dtype=np.uint8)
        out_n_valid = np.empty(n_groups, dtype=np.int64)
        out_n_filtered = np.empty(n_groups, dtype=np.int64)
        out_cond = np.empty(n_groups, dtype=np.float64)
        
        fit_groups_multifit_numba(
            X_all, Y_all, W_all, offsets,
            n_groups, n_feat, n_targets, n_params,
            fit_intercept, min_stat, compute_mad,
            out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond,
        )
        
        return (out_beta, out_errors, out_rms, out_mad,
                out_status, out_n_valid, out_n_filtered, out_cond)
    
    else:
        # Single-fit kernel (one target at a time)
        if n_targets == 1:
            # Simple case: single target
            Y_1d = Y_all.ravel() if Y_all.ndim > 1 else Y_all
            
            out_beta = np.empty((n_groups, n_params), dtype=np.float64)
            out_errors = np.empty((n_groups, n_params), dtype=np.float64)
            out_rms = np.empty(n_groups, dtype=np.float64)
            out_mad = np.empty(n_groups, dtype=np.float64)
            out_status = np.empty(n_groups, dtype=np.uint8)
            out_n_valid = np.empty(n_groups, dtype=np.int64)
            out_n_filtered = np.empty(n_groups, dtype=np.int64)
            out_cond = np.empty(n_groups, dtype=np.float64)
            out_sum_y = np.empty(n_groups, dtype=np.float64)
            out_sum_y2 = np.empty(n_groups, dtype=np.float64)
            
            fit_groups_single_numba(
                X_all, Y_1d, W_all, offsets,
                n_groups, n_feat, n_params,
                fit_intercept, min_stat, compute_mad, invalid_handling,
                out_beta, out_errors, out_rms, out_mad,
                out_status, out_n_valid, out_n_filtered, out_cond,
                out_sum_y, out_sum_y2,
            )
            
            return (out_beta, out_errors, out_rms, out_mad,
                    out_status, out_n_valid, out_n_filtered, out_cond)
        
        else:
            # Multiple targets with filtering: call single-fit for each target
            if Y_all.ndim == 1:
                Y_all = Y_all.reshape(-1, 1)
            
            out_beta = np.empty((n_groups, n_targets, n_params), dtype=np.float64)
            out_errors = np.empty((n_groups, n_targets, n_params), dtype=np.float64)
            out_rms = np.empty((n_groups, n_targets), dtype=np.float64)
            out_mad = np.empty((n_groups, n_targets), dtype=np.float64)
            out_status = np.empty((n_groups, n_targets), dtype=np.uint8)
            out_n_valid = np.empty((n_groups, n_targets), dtype=np.int64)
            out_n_filtered = np.empty((n_groups, n_targets), dtype=np.int64)
            out_cond = np.empty((n_groups, n_targets), dtype=np.float64)
            
            for t in range(n_targets):
                Y_t = Y_all[:, t]
                
                beta_t = np.empty((n_groups, n_params), dtype=np.float64)
                errors_t = np.empty((n_groups, n_params), dtype=np.float64)
                rms_t = np.empty(n_groups, dtype=np.float64)
                mad_t = np.empty(n_groups, dtype=np.float64)
                status_t = np.empty(n_groups, dtype=np.uint8)
                n_valid_t = np.empty(n_groups, dtype=np.int64)
                n_filtered_t = np.empty(n_groups, dtype=np.int64)
                cond_t = np.empty(n_groups, dtype=np.float64)
                sum_y_t = np.empty(n_groups, dtype=np.float64)
                sum_y2_t = np.empty(n_groups, dtype=np.float64)
                
                fit_groups_single_numba(
                    X_all, Y_t, W_all, offsets,
                    n_groups, n_feat, n_params,
                    fit_intercept, min_stat, compute_mad, invalid_handling,
                    beta_t, errors_t, rms_t, mad_t,
                    status_t, n_valid_t, n_filtered_t, cond_t,
                    sum_y_t, sum_y2_t,
                )
                
                out_beta[:, t, :] = beta_t
                out_errors[:, t, :] = errors_t
                out_rms[:, t] = rms_t
                out_mad[:, t] = mad_t
                out_status[:, t] = status_t
                out_n_valid[:, t] = n_valid_t
                out_n_filtered[:, t] = n_filtered_t
                out_cond[:, t] = cond_t
            
            return (out_beta, out_errors, out_rms, out_mad,
                    out_status, out_n_valid, out_n_filtered, out_cond)


# ============================================================================
# NUMPY FALLBACK (For when Numba is not available)
# ============================================================================

def _fit_groups_numpy_fallback(
    X_all, Y_all, W_all, offsets,
    n_groups, n_feat, n_params,
    fit_intercept, min_stat, compute_mad,
    invalid_handling, n_targets,
):
    """NumPy-based fallback when Numba is not available."""
    # Allocate outputs
    if n_targets == 1:
        out_beta = np.full((n_groups, n_params), np.nan)
        out_errors = np.full((n_groups, n_params), np.nan)
        out_rms = np.full(n_groups, np.nan)
        out_mad = np.full(n_groups, np.nan)
        out_status = np.zeros(n_groups, dtype=np.uint8)
        out_n_valid = np.zeros(n_groups, dtype=np.int64)
        out_n_filtered = np.zeros(n_groups, dtype=np.int64)
        out_cond = np.full(n_groups, np.nan)
    else:
        out_beta = np.full((n_groups, n_targets, n_params), np.nan)
        out_errors = np.full((n_groups, n_targets, n_params), np.nan)
        out_rms = np.full((n_groups, n_targets), np.nan)
        out_mad = np.full((n_groups, n_targets), np.nan)
        out_status = np.zeros((n_groups, n_targets), dtype=np.uint8)
        out_n_valid = np.zeros((n_groups, n_targets), dtype=np.int64)
        out_n_filtered = np.zeros((n_groups, n_targets), dtype=np.int64)
        out_cond = np.full((n_groups, n_targets), np.nan)
    
    # Simple NumPy implementation for each group
    for gi in range(n_groups):
        i0 = offsets[gi]
        i1 = offsets[gi + 1]
        
        X_slice = X_all[i0:i1]
        
        if n_targets == 1:
            Y_slice = Y_all[i0:i1] if Y_all.ndim == 1 else Y_all[i0:i1, 0]
            
            # Build design matrix
            if fit_intercept:
                X_design = np.column_stack([np.ones(len(X_slice)), X_slice])
            else:
                X_design = X_slice
            
            # Simple OLS
            try:
                coeffs, residuals, rank, s = np.linalg.lstsq(X_design, Y_slice, rcond=None)
                out_beta[gi] = coeffs
                out_n_valid[gi] = len(X_slice)
                
                # Compute RMS
                y_pred = X_design @ coeffs
                resid = Y_slice - y_pred
                out_rms[gi] = np.sqrt(np.mean(resid ** 2))
                
                if compute_mad:
                    out_mad[gi] = np.median(np.abs(resid - np.median(resid)))
                    
            except np.linalg.LinAlgError:
                out_status[gi] = STATUS_NUMERICAL_ERROR
        else:
            for t in range(n_targets):
                Y_slice = Y_all[i0:i1, t] if Y_all.ndim > 1 else Y_all[i0:i1]
                
                if fit_intercept:
                    X_design = np.column_stack([np.ones(len(X_slice)), X_slice])
                else:
                    X_design = X_slice
                
                try:
                    coeffs, residuals, rank, s = np.linalg.lstsq(X_design, Y_slice, rcond=None)
                    out_beta[gi, t] = coeffs
                    out_n_valid[gi, t] = len(X_slice)
                    
                    y_pred = X_design @ coeffs
                    resid = Y_slice - y_pred
                    out_rms[gi, t] = np.sqrt(np.mean(resid ** 2))
                    
                    if compute_mad:
                        out_mad[gi, t] = np.median(np.abs(resid - np.median(resid)))
                        
                except np.linalg.LinAlgError:
                    out_status[gi, t] = STATUS_NUMERICAL_ERROR
    
    return (out_beta, out_errors, out_rms, out_mad,
            out_status, out_n_valid, out_n_filtered, out_cond)


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

def _warmup_kernels():
    """Trigger JIT compilation with minimal data."""
    if not _NUMBA_AVAILABLE:
        return
    
    # Small test data
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    Y = np.array([1.0, 2.0, 3.0, 4.0])
    Y_multi = np.column_stack([Y, Y * 2])
    W = np.empty(0)
    offsets = np.array([0, 4], dtype=np.int64)
    
    # Allocate outputs for single-fit
    out_beta = np.empty((1, 3), dtype=np.float64)
    out_errors = np.empty((1, 3), dtype=np.float64)
    out_rms = np.empty(1, dtype=np.float64)
    out_mad = np.empty(1, dtype=np.float64)
    out_status = np.empty(1, dtype=np.uint8)
    out_n_valid = np.empty(1, dtype=np.int64)
    out_n_filtered = np.empty(1, dtype=np.int64)
    out_cond = np.empty(1, dtype=np.float64)
    out_sum_y = np.empty(1, dtype=np.float64)
    out_sum_y2 = np.empty(1, dtype=np.float64)
    
    # Warmup single-fit
    fit_groups_single_numba(
        X, Y, W, offsets,
        1, 2, 3, True, 2, True, INVALID_DETECT,
        out_beta, out_errors, out_rms, out_mad,
        out_status, out_n_valid, out_n_filtered, out_cond,
        out_sum_y, out_sum_y2,
    )
    
    # Allocate outputs for multi-fit
    out_beta_m = np.empty((1, 2, 3), dtype=np.float64)
    out_errors_m = np.empty((1, 2, 3), dtype=np.float64)
    out_rms_m = np.empty((1, 2), dtype=np.float64)
    out_mad_m = np.empty((1, 2), dtype=np.float64)
    out_status_m = np.empty((1, 2), dtype=np.uint8)
    
    # Warmup multi-fit
    fit_groups_multifit_numba(
        X, Y_multi, W, offsets,
        1, 2, 2, 3, True, 2, True,
        out_beta_m, out_errors_m, out_rms_m, out_mad_m,
        out_status_m, out_n_valid, out_n_filtered, out_cond,
    )


# Trigger JIT compilation on module import
if _NUMBA_AVAILABLE:
    _warmup_kernels()
