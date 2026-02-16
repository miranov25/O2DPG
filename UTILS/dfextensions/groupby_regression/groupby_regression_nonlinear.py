"""
Non-Linear Sliding Window Fit

Phase 13.10.GB — standalone entry point for non-linear fits within
sliding windows.  Imports windowing infrastructure (neighbor enumeration,
boundary handling, data collection) from groupby_regression_sliding_window.py
but does NOT modify that file.

Entry point: make_nonlinear_sliding_window_fit()

Author: Claude13 (Implementer)
"""

import sys
import time
import numpy as np
import pandas as pd
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
#  Import windowing utilities from existing SW module
# ---------------------------------------------------------------------------

from .groupby_regression_sliding_window import (
    _validate_sliding_window_inputs,
    _build_bin_index_map,
    _generate_neighbor_offsets,
    _resolve_boundary,
    _validate_periodic_dims,
    _resolve_kernel_width,
    _precompute_offset_weights,
    _aggregate_window_zerocopy,
    _AggResult,
    _get_neighbor_bins,
)

from .groupby_regression_models import get_model, ModelSpec

# ---------------------------------------------------------------------------
#  scipy availability
# ---------------------------------------------------------------------------

_SCIPY_AVAILABLE = False
try:
    from scipy.optimize import curve_fit as _curve_fit
    _SCIPY_AVAILABLE = True
except ImportError:
    _curve_fit = None


# ===================================================================== #
#  Public API
# ===================================================================== #

def make_nonlinear_sliding_window_fit(
    *,
    df: pd.DataFrame,
    gb_columns: List[str],
    fit_columns: List[str],
    linear_columns: List[str],
    window_spec: Dict[str, int],
    fit_func: Union[str, Callable],
    weights: Optional[str] = None,
    suffix: str = '_nl',
    selection: Optional[pd.Series] = None,
    min_stat: int = 10,
    cast_dtype: str = 'float64',
    boundary: Union[str, Dict[str, str]] = 'full',
    kernel: Union[str, Callable] = 'uniform',
    kernel_width: Optional[Union[float, Dict[str, float]]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    return_metadata: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, dict]]:
    """
    Non-linear sliding window fit.

    For each bin defined by gb_columns, aggregates data from neighboring bins
    and calls a user-provided callable or named model to fit the data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    gb_columns : list of str
        Columns defining the bin grid (integer-valued).
    fit_columns : list of str
        Target columns for fitting.
    linear_columns : list of str
        Predictor columns (X matrix).  For non-linear fits these serve as
        the independent variable(s) passed to the fit function.
    window_spec : dict[str, int]
        Half-width of the sliding window per dimension.
    fit_func : str or callable
        Named model string (e.g. ``'gaussian'``) or custom callable with
        signature ``fit_func(X, y, weights, **kwargs) → (coefficients, diagnostics)``.
        See Notes for the callable contract.
    weights : str, optional
        Column name for sample weights (inverse-variance convention: w = 1/σ²).
    suffix : str
        Suffix appended to output column names (default: ``'_nl'``).
    selection : pd.Series, optional
        Boolean mask to filter input rows.
    min_stat : int
        Minimum valid rows per window to attempt fit.
    cast_dtype : str
        Output dtype for float columns.
    boundary : str or dict
        Boundary handling: ``'full'``, ``'truncate'``, ``'symmetric'``, ``'periodic'``.
    kernel : str or callable
        Distance kernel: ``'uniform'``, ``'gaussian'``, ``'epanechnikov'``, ``'linear'``.
    kernel_width : float or dict, optional
        Kernel bandwidth per dimension.
    fit_kwargs : dict, optional
        Forwarded as ``**kwargs`` to custom callables.
    optimizer_kwargs : dict, optional
        Passed to ``scipy.optimize.curve_fit`` for named models.
        Common keys: ``p0`` (initial params), ``bounds``, ``maxfev``.
        If ``p0`` is not provided and the model has an ``estimate_p0``
        function, initial parameters are estimated from data per bin.
    return_metadata : bool
        If True, return ``(DataFrame, metadata_dict)``.
    verbose : bool
        Print progress information.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, dict)
        Group-level fit results with columns for coefficients, errors,
        diagnostics, and window metadata.

    Notes
    -----
    **Custom callable contract:**

    The callable receives ``(X, y, weights, **fit_kwargs)`` and must return
    ``(coefficients, diagnostics)`` where both are ``dict[str, float]``.

    - ``X``: shape ``(n_samples, n_predictors)`` — no intercept column added
    - ``y``: shape ``(n_samples,)`` — one target at a time
    - ``weights``: shape ``(n_samples,)`` — always provided (uniform 1.0 if unspecified)
    - ``coefficients``: keys become column names ``{target}_{key}{suffix}``
    - ``diagnostics``: keys become column names ``{target}_{key}{suffix}``
    - Any exception → bin flagged as failed, NaN output

    **Named model p0 resolution order:**

    1. Explicit ``optimizer_kwargs['p0']`` (global, same for all bins)
    2. Model's ``estimate_p0(x, y, weights)`` (per-bin, data-driven)
    3. Model's ``default_p0`` (global fallback)
    """
    t0 = time.time()

    # ---- Resolve fit function ----
    if isinstance(fit_func, str):
        resolved_func, param_names, model_name = _resolve_named_model(
            fit_func, fit_kwargs, optimizer_kwargs)
    elif callable(fit_func):
        resolved_func = fit_func
        param_names = []  # discovered from first successful fit
        model_name = '<callable>'
    else:
        raise TypeError(
            f"fit_func must be a callable or named model string, got {type(fit_func)}")

    # ---- Window setup (reused from linear SW) ----
    full_window_spec = {dim: window_spec.get(dim, 0) for dim in gb_columns}

    _validate_sliding_window_inputs(
        df=df, gb_columns=gb_columns, window_spec=full_window_spec,
        fit_columns=fit_columns, linear_columns=linear_columns,
        weights=weights, selection=selection, min_stat=min_stat,
        backend='numpy',
    )

    neighbor_offsets = _generate_neighbor_offsets(full_window_spec, gb_columns)

    _sel_df = df if selection is None else df[selection]
    bounds = {dim: (int(_sel_df[dim].min()), int(_sel_df[dim].max()))
              for dim in gb_columns}

    boundary_resolved = _resolve_boundary(boundary, gb_columns)
    _validate_periodic_dims(boundary_resolved, bounds, full_window_spec)

    _is_weighted_kernel = (kernel != 'uniform') if isinstance(kernel, str) else True
    kernel_width_resolved = _resolve_kernel_width(kernel_width, full_window_spec, gb_columns)

    if verbose:
        print(f"[NL-SW] fit_model={model_name}, bins={len(gb_columns)}D, "
              f"window={full_window_spec}")

    # ---- Build bin map and aggregate (V1 recompute path) ----
    bin_map = _build_bin_index_map(df, gb_columns, selection)
    center_bins = list(bin_map.keys())

    agg_results = _aggregate_window_zerocopy(
        df=df, bin_map=bin_map, center_bins=center_bins,
        neighbor_offsets=neighbor_offsets, bounds=bounds,
        gb_columns=gb_columns, fit_columns=fit_columns,
        weights=weights,
    )

    if verbose:
        print(f"[NL-SW] Aggregation done: {len(agg_results)} bins, "
              f"{time.time()-t0:.3f}s elapsed")

    # ---- Non-linear fit dispatch ----
    fit_results = _fit_nonlinear(
        df=df, agg_results=agg_results,
        fit_columns=fit_columns, linear_columns=linear_columns,
        weights=weights, min_stat=min_stat,
        fit_func=resolved_func, fit_kwargs=fit_kwargs,
    )

    if verbose:
        print(f"[NL-SW] Fitting done: {time.time()-t0:.3f}s elapsed")

    # ---- Discover param_names from results (custom callables) ----
    if not param_names:
        param_names = _discover_param_names(fit_results)

    # ---- Assemble output ----
    out = _assemble_nonlinear_results(
        gb_columns=gb_columns, agg_results=agg_results,
        fit_results=fit_results, fit_columns=fit_columns,
    )

    # Apply suffix
    if suffix:
        bin_cols = set(gb_columns)
        rename_map = {c: c + suffix for c in out.columns if c not in bin_cols}
        out = out.rename(columns=rename_map)

    # Cast dtype
    if cast_dtype:
        float_cols = out.select_dtypes(include=[np.floating]).columns
        if len(float_cols) > 0:
            out[float_cols] = out[float_cols].astype(cast_dtype)

    # ---- Metadata ----
    metadata = _build_nonlinear_metadata(
        fit_columns=fit_columns, linear_columns=linear_columns,
        suffix=suffix, gb_columns=gb_columns,
        model_name=model_name, param_names=param_names,
        fit_kwargs=fit_kwargs, optimizer_kwargs=optimizer_kwargs,
        min_stat=min_stat, weights_column=weights,
        window_spec=full_window_spec,
        boundary_mode={dim: boundary_resolved[dim] for dim in gb_columns},
        kernel=kernel, kernel_width=kernel_width_resolved,
        n_bins=len(center_bins),
        computation_time_sec=time.time() - t0,
    )
    out.attrs.update(metadata)

    if verbose:
        print(f"[NL-SW] Complete: {len(out)} bins, {time.time()-t0:.3f}s total")

    if return_metadata:
        return out, metadata
    return out


# ===================================================================== #
#  Named model → callable wrapper
# ===================================================================== #

def _resolve_named_model(
    model_name: str,
    fit_kwargs: Optional[Dict[str, Any]],
    optimizer_kwargs: Optional[Dict[str, Any]],
) -> Tuple[Callable, List[str], str]:
    """Wrap a named model into a callable with the fit_func interface.

    Returns (callable, param_names, model_name).

    p0 resolution order:
      1. optimizer_kwargs['p0']   — explicit global
      2. spec.estimate_p0(x,y,w)  — per-bin data-driven
      3. spec.default_p0          — global fallback
    """
    if not _SCIPY_AVAILABLE:
        raise ImportError(
            "scipy is required for named models. "
            "Install with: pip install scipy>=1.7.0"
        )

    spec = get_model(model_name)
    opt_kw = dict(optimizer_kwargs or {})

    # Apply model defaults where user didn't specify
    has_explicit_p0 = 'p0' in opt_kw
    if 'bounds' not in opt_kw and spec.default_bounds is not None:
        opt_kw['bounds'] = spec.default_bounds
    if 'maxfev' not in opt_kw:
        opt_kw['maxfev'] = 10000

    param_names = spec.param_names

    def _named_model_fit(X, y, weights, **user_kwargs):
        """Wrapper: calls scipy.optimize.curve_fit on the named model."""
        # X → 1D predictor for curve_fit
        if X.ndim == 2 and X.shape[1] == 1:
            x = X[:, 0]
        elif X.ndim == 2 and X.shape[1] > 1:
            raise ValueError(
                f"Named model '{model_name}' requires 1 predictor "
                f"(got {X.shape[1]}). Use a custom callable for multi-predictor.")
        else:
            x = X

        # p0 resolution: explicit → estimate → default
        local_kw = dict(opt_kw)
        if not has_explicit_p0:
            if spec.estimate_p0 is not None:
                try:
                    local_kw['p0'] = spec.estimate_p0(x, y, weights)
                except Exception:
                    if spec.default_p0 is not None:
                        local_kw['p0'] = spec.default_p0
            elif spec.default_p0 is not None:
                local_kw['p0'] = spec.default_p0

        # Weights → sigma for curve_fit
        # Convention: weights = 1/σ² → sigma = 1/sqrt(weights)  (AC-1)
        sigma = None
        absolute_sigma = False
        if weights is not None and not np.all(weights == 1.0):
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma = np.where(weights > 0, 1.0 / np.sqrt(weights), np.inf)
            absolute_sigma = True

        try:
            popt, pcov = _curve_fit(
                spec.func, x, y,
                sigma=sigma, absolute_sigma=absolute_sigma,
                **local_kw,
            )

            coefficients = {
                param_names[i]: float(popt[i])
                for i in range(len(param_names))
            }

            # Standard errors from covariance diagonal
            if pcov is not None and np.isfinite(pcov).all():
                perr = np.sqrt(np.diag(pcov))
            else:
                perr = np.full(len(param_names), np.nan)

            for i, pn in enumerate(param_names):
                coefficients[f'{pn}_err'] = float(perr[i])

            # Diagnostics
            y_pred = spec.func(x, *popt)
            residuals = y - y_pred
            chi2 = float(np.sum(residuals ** 2))
            ndf = max(len(y) - len(param_names), 1)

            diagnostics = {
                'chi2': chi2,
                'ndf': float(ndf),
                'chi2_ndf': chi2 / ndf if ndf > 0 else np.nan,
                'converged': 1.0,
                'n_fitted': float(len(y)),
            }
            return coefficients, diagnostics

        except (RuntimeError, ValueError, TypeError) as exc:
            coefficients = {pn: np.nan for pn in param_names}
            for pn in param_names:
                coefficients[f'{pn}_err'] = np.nan
            diagnostics = {
                'chi2': np.nan, 'ndf': np.nan, 'chi2_ndf': np.nan,
                'converged': 0.0, 'n_fitted': float(len(y)),
            }
            return coefficients, diagnostics

    return _named_model_fit, param_names, model_name


# ===================================================================== #
#  Non-linear fit dispatch (V1 recompute path + callable)
# ===================================================================== #

def _fit_nonlinear(
    df: pd.DataFrame,
    agg_results: List[_AggResult],
    fit_columns: List[str],
    linear_columns: List[str],
    weights: Optional[str],
    min_stat: int,
    fit_func: Callable,
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]]:
    """Dispatch non-linear fit per center bin, per target.

    Uses pre-aggregated row indices from _aggregate_window_zerocopy.
    """
    out = {}
    fkw = dict(fit_kwargs or {})

    # Pre-extract arrays ONCE
    pred_arrays = {p: df[p].to_numpy(dtype=np.float64) for p in linear_columns}
    target_arrays = {t: df[t].to_numpy(dtype=np.float64) for t in fit_columns}
    w_array = df[weights].to_numpy(dtype=np.float64) if weights is not None else None

    for ar in agg_results:
        center_map = {}

        if ar.row_indices.size == 0:
            for t in fit_columns:
                center_map[t] = _empty_result("empty_window", 0)
            out[ar.center] = center_map
            continue

        idx = ar.row_indices
        X_cols = [pred_arrays[p][idx] for p in linear_columns]

        for t in fit_columns:
            y = target_arrays[t][idx]

            # NaN filtering
            valid = np.isfinite(y)
            for xc in X_cols:
                valid &= np.isfinite(xc)
            if w_array is not None:
                w = w_array[idx]
                valid &= np.isfinite(w) & (w > 0)

            n_valid = int(np.sum(valid))
            if n_valid < max(1, int(min_stat)):
                center_map[t] = _empty_result("insufficient_stats", n_valid)
                continue

            y_v = y[valid]
            if len(linear_columns) > 0:
                X_v = np.column_stack([xc[valid] for xc in X_cols])
            else:
                X_v = np.empty((n_valid, 0), dtype=np.float64)

            w_v = (w_array[idx][valid] if w_array is not None
                   else np.ones(n_valid, dtype=np.float64))

            try:
                coefficients, diagnostics = fit_func(X_v, y_v, w_v, **fkw)
            except Exception as exc:
                center_map[t] = _empty_result(
                    f"fit_failed: {type(exc).__name__}: {exc}", n_valid)
                continue

            if not isinstance(coefficients, dict) or not isinstance(diagnostics, dict):
                center_map[t] = _empty_result(
                    "bad_return_type: must return (dict, dict)", n_valid)
                continue

            # Separate coefficient keys from _err keys
            coeff_keys = [k for k in coefficients if not k.endswith('_err')]
            err_keys = [k for k in coefficients if k.endswith('_err')]

            center_map[t] = {
                'coefficients': {k: float(coefficients[k]) for k in coeff_keys},
                'errors': {k: float(coefficients[k]) for k in err_keys},
                'diagnostics': {k: float(v) for k, v in diagnostics.items()},
                'n_fitted': int(diagnostics.get('n_fitted', n_valid)),
                'quality_flag': ('' if diagnostics.get('converged', 1.0) > 0.5
                                 else 'fit_failed'),
            }

        out[ar.center] = center_map
    return out


def _empty_result(reason: str, n_available: int) -> Dict[str, Any]:
    """Empty non-linear fit result."""
    return {
        'coefficients': {}, 'errors': {}, 'diagnostics': {},
        'n_fitted': n_available, 'quality_flag': reason,
    }


def _discover_param_names(
    fit_results: Dict[Tuple, Dict[str, Dict[str, Any]]],
) -> List[str]:
    """Extract parameter names from the first successful fit result."""
    for center, tmap in fit_results.items():
        for t, res in tmap.items():
            names = list(res.get('coefficients', {}).keys())
            if names:
                return names
    return []


# ===================================================================== #
#  Result assembly
# ===================================================================== #

def _assemble_nonlinear_results(
    gb_columns: List[str],
    agg_results: List[_AggResult],
    fit_results: Dict[Tuple[int, ...], Dict[str, Dict[str, Any]]],
    fit_columns: List[str],
) -> pd.DataFrame:
    """Assemble non-linear fit results into a DataFrame.

    Output columns: gb_columns + window metadata + per-target coefficients,
    errors, diagnostics, n_fitted, quality_flag.
    """
    # Discover all unique keys across results
    all_coeff_keys = {t: set() for t in fit_columns}
    all_err_keys = {t: set() for t in fit_columns}
    all_diag_keys = {t: set() for t in fit_columns}

    for center, target_map in fit_results.items():
        for t in fit_columns:
            if t not in target_map:
                continue
            res = target_map[t]
            all_coeff_keys[t].update(res.get('coefficients', {}).keys())
            all_err_keys[t].update(res.get('errors', {}).keys())
            all_diag_keys[t].update(res.get('diagnostics', {}).keys())

    for t in fit_columns:
        all_coeff_keys[t] = sorted(all_coeff_keys[t])
        all_err_keys[t] = sorted(all_err_keys[t])
        all_diag_keys[t] = sorted(all_diag_keys[t])

    rows = []
    for ar in agg_results:
        row = {}

        # Bin coordinates
        for i, col in enumerate(gb_columns):
            row[col] = ar.center[i]

        # Window metadata
        row['n_neighbors_used'] = ar.n_neighbors_used
        row['n_rows_aggregated'] = ar.n_rows_aggregated
        row['effective_window_fraction'] = ar.effective_window_fraction

        # Per-target aggregation stats
        for t in fit_columns:
            st = ar.stats.get(t, {})
            row[f'{t}_mean'] = st.get('mean', np.nan)
            row[f'{t}_std'] = st.get('std', np.nan)
            row[f'{t}_entries'] = st.get('entries', 0)

        # Per-target fit results
        target_map = fit_results.get(ar.center, {})
        for t in fit_columns:
            res = target_map.get(t, _empty_result('missing', 0))
            coeffs = res.get('coefficients', {})
            errs = res.get('errors', {})
            diags = res.get('diagnostics', {})

            for k in all_coeff_keys[t]:
                row[f'{t}_{k}'] = coeffs.get(k, np.nan)
            for k in all_err_keys[t]:
                row[f'{t}_{k}'] = errs.get(k, np.nan)
            for k in all_diag_keys[t]:
                row[f'{t}_{k}'] = diags.get(k, np.nan)

            row[f'{t}_n_fitted'] = res.get('n_fitted', 0)
            row[f'{t}_quality_flag'] = res.get('quality_flag', '')

        rows.append(row)

    return pd.DataFrame(rows)


# ===================================================================== #
#  Metadata
# ===================================================================== #

def _build_nonlinear_metadata(
    fit_columns, linear_columns, suffix, gb_columns,
    model_name, param_names,
    fit_kwargs=None, optimizer_kwargs=None,
    min_stat=10, weights_column=None,
    window_spec=None, boundary_mode=None,
    kernel='uniform', kernel_width=None,
    n_bins=0, computation_time_sec=0.0,
) -> Dict[str, Any]:
    """Build V4-compatible metadata for non-linear SW results."""
    coefficients_cols = {}
    errors_cols = {}
    quality_cols = {}

    for target in fit_columns:
        coefficients_cols[target] = [f'{target}_{p}{suffix}' for p in param_names]
        errors_cols[target] = [f'{target}_{p}_err{suffix}' for p in param_names]
        quality_cols[target] = [
            f'{target}_chi2{suffix}',
            f'{target}_ndf{suffix}',
            f'{target}_converged{suffix}',
        ]

    metadata = {
        'version': '1.1',
        'formulas': {},
        'residual_formulas': {},
        'pull_formulas': {},
        'columns': {
            'gb_columns': list(gb_columns),
            'fit_columns': list(fit_columns),
            'linear_columns': list(linear_columns),
            'coefficients': coefficients_cols,
            'errors': errors_cols,
            'quality': quality_cols,
        },
        'parameters': {
            'suffix': suffix,
            'fit_intercept': None,
            'min_stat': min_stat,
            'fit_type': 'sliding_window',
            'fit_model': model_name,
            'param_names': list(param_names),
            'weights_column': weights_column,
        },
        # SW-specific flat keys
        'window_spec': window_spec or {},
        'boundary_mode': boundary_mode or {},
        'kernel': kernel if isinstance(kernel, str) else 'custom',
        'kernel_width': kernel_width,
        'algorithm': 'recompute',
        'backend_used': 'nonlinear_numpy',
        'n_bins': n_bins,
        'computation_time_sec': computation_time_sec,
        'python_version': sys.version,
    }

    if optimizer_kwargs is not None:
        metadata['parameters']['optimizer_kwargs'] = _serialize_optimizer_kwargs(
            optimizer_kwargs)

    return metadata


def _serialize_optimizer_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Make optimizer_kwargs JSON-safe."""
    result = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        elif isinstance(v, (list, tuple)):
            result[k] = [
                x.tolist() if isinstance(x, np.ndarray) else x for x in v
            ]
        else:
            result[k] = v
    return result
