"""
GroupBy Regression Evaluator & Interpolator

Phase 13.9.GB — Evaluate and interpolate GroupBy regression coefficient maps.

Wraps regression output (dfGB or dict-of-arrays) into a structured evaluator
supporting per-bin lookup, multilinear interpolation, inverse-variance weighting,
and JSON export for browser-side evaluation.

Design decisions (unanimous reviewer consensus, 2026-02-15):
  - Multi-target: targets=['dX','dY','dZ'] with shared grid
  - Dual constructor: __init__(arrays) + from_dfGB(DataFrame)
  - Dict-of-arrays core for C++/JS portability
  - Suffix stripping in from_dfGB, internal keys are canonical
  - bounds='clamp' default (configurable 'clamp'|'nan'|'extrapolate')
  - compare() deferred to Phase 13.10

Author: Claude13 (Main Reviewer) + Claude14 (Coder)
"""

import itertools
import json
import gzip
import numpy as np
import warnings
from typing import (
    Any, Callable, Dict, List, Optional, Sequence, Tuple, Union,
)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class GroupByRegressionEvaluator:
    """
    Evaluate and interpolate GroupBy regression coefficient maps.

    Supports multi-target evaluation (e.g., dX/dY/dZ) on a shared
    N-dimensional bin grid with configurable interpolation, sparse
    grid handling, and JSON export for browser-side evaluation.

    Parameters
    ----------
    grid_shape : tuple of int
        Shape of the coefficient grid (n_bins per dimension).
    group_columns : list of str
        Bin dimension names, e.g. ['xBin', 'yBin', 'zBin'].
    predictor_columns : list of str
        Predictor names, e.g. ['spaceCharge'].
    targets : list of str
        Target variable names, e.g. ['dX', 'dY', 'dZ'].
    bin_centers : dict[str, np.ndarray]
        Physical coordinates for each dimension. Keys must match
        group_columns. Arrays must be sorted and have length matching
        the corresponding grid_shape dimension.
    coefficients : dict[str, dict[str, np.ndarray]]
        Nested dict: {target: {coeff_name: N-D array}}.
        Coefficient names are canonical (no suffix):
        'intercept', 'slope_{pred}', 'intercept_err', 'slope_{pred}_err',
        'rms'/'rmse', 'r_squared', 'std', 'median', 'mad', 'n_fitted'.
    valid_mask : np.ndarray of bool
        N-D boolean array (shape=grid_shape). True where fit is valid.
        If None, inferred from NaN in intercept of first target.
    metadata : dict, optional
        Additional metadata (source function, creation date, etc.).

    Examples
    --------
    >>> # From arrays (portable — works in C++/JS too)
    >>> evaluator = GroupByRegressionEvaluator(
    ...     grid_shape=(10, 20),
    ...     group_columns=['xBin', 'yBin'],
    ...     predictor_columns=['meanIDC'],
    ...     targets=['dX'],
    ...     bin_centers={'xBin': np.arange(10, dtype=float),
    ...                  'yBin': np.arange(20, dtype=float)},
    ...     coefficients={'dX': {
    ...         'intercept': np.random.randn(10, 20),
    ...         'slope_meanIDC': np.random.randn(10, 20),
    ...     }},
    ... )
    >>> result = evaluator.evaluate({'xBin': 3.5, 'yBin': 10.2},
    ...                              {'meanIDC': 150.0})

    >>> # From DataFrame (convenience)
    >>> evaluator = GroupByRegressionEvaluator.from_dfGB(
    ...     dfGB, group_columns=['xBin', 'yBin'],
    ...     predictor_columns=['meanIDC'], targets=['dX'],
    ...     suffix='_sw',
    ... )
    """

    # Schema version for JSON export compatibility
    SCHEMA_VERSION = "1.0"

    # Known coefficient keys that the evaluator recognises
    _COEFF_KEYS = {'intercept', 'intercept_err', 'rms', 'rmse', 'mad',
                   'r_squared', 'std', 'median', 'n_fitted'}
    # slope_* and slope_*_err are dynamic — matched by prefix

    def __init__(
        self,
        grid_shape: Tuple[int, ...],
        group_columns: List[str],
        predictor_columns: List[str],
        targets: Union[str, List[str]],
        bin_centers: Dict[str, np.ndarray],
        coefficients: Dict[str, Dict[str, np.ndarray]],
        valid_mask: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        # Normalise targets to list
        if isinstance(targets, str):
            targets = [targets]
        self._targets = list(targets)
        self._group_columns = list(group_columns)
        self._predictor_columns = list(predictor_columns)
        self._grid_shape = tuple(grid_shape)
        self._metadata = dict(metadata) if metadata else {}

        # Detect fit model type from metadata
        params_info = self._metadata.get('parameters', {})
        self._fit_model = params_info.get('fit_model', 'linear')
        self._param_names = params_info.get('param_names', [])
        self._is_nonlinear = (self._fit_model != 'linear')

        # Validate bin_centers
        if set(bin_centers.keys()) != set(group_columns):
            raise ValueError(
                f"bin_centers keys {set(bin_centers.keys())} must match "
                f"group_columns {set(group_columns)}")
        self._bin_centers = {}
        for i, col in enumerate(group_columns):
            arr = np.asarray(bin_centers[col], dtype=np.float64)
            if len(arr) != grid_shape[i]:
                raise ValueError(
                    f"bin_centers['{col}'] length {len(arr)} != "
                    f"grid_shape[{i}]={grid_shape[i]}")
            self._bin_centers[col] = arr

        # Validate coefficients
        if set(coefficients.keys()) != set(self._targets):
            raise ValueError(
                f"coefficients keys {set(coefficients.keys())} must match "
                f"targets {set(self._targets)}")
        self._coefficients = {}
        for tgt in self._targets:
            tgt_coeffs = {}
            for key, arr in coefficients[tgt].items():
                arr = np.asarray(arr, dtype=np.float64)
                if arr.shape != self._grid_shape:
                    raise ValueError(
                        f"coefficients['{tgt}']['{key}'] shape {arr.shape} "
                        f"!= grid_shape {self._grid_shape}")
                tgt_coeffs[key] = arr

            if self._is_nonlinear:
                # Non-linear: require at least one parameter grid
                if len(tgt_coeffs) == 0:
                    raise ValueError(
                        f"coefficients['{tgt}'] has no entries")
            else:
                # Linear: require intercept + slope_* (original behavior)
                if 'intercept' not in tgt_coeffs:
                    raise ValueError(
                        f"coefficients['{tgt}'] must contain 'intercept'")
                for pred in predictor_columns:
                    if f'slope_{pred}' not in tgt_coeffs:
                        raise ValueError(
                            f"coefficients['{tgt}'] must contain "
                            f"'slope_{pred}' for predictor '{pred}'")
            self._coefficients[tgt] = tgt_coeffs

        # Valid mask
        if valid_mask is not None:
            self._valid_mask = np.asarray(valid_mask, dtype=bool)
            if self._valid_mask.shape != self._grid_shape:
                raise ValueError(
                    f"valid_mask shape {self._valid_mask.shape} != "
                    f"grid_shape {self._grid_shape}")
        else:
            # Infer from first coefficient of first target
            first_tgt = self._targets[0]
            first_key = next(iter(self._coefficients[first_tgt]))
            self._valid_mask = ~np.isnan(
                self._coefficients[first_tgt][first_key])

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        """Shape of the coefficient grid as tuple."""
        return self._grid_shape

    @property
    def group_columns(self) -> List[str]:
        """Bin dimension names."""
        return list(self._group_columns)

    @property
    def predictor_columns(self) -> List[str]:
        """Predictor variable names."""
        return list(self._predictor_columns)

    @property
    def targets(self) -> List[str]:
        """Target variable names."""
        return list(self._targets)

    @property
    def dimensions(self) -> Dict[str, Dict[str, float]]:
        """Dimension names with coordinate ranges."""
        return {
            col: {'min': float(self._bin_centers[col][0]),
                   'max': float(self._bin_centers[col][-1]),
                   'n_bins': len(self._bin_centers[col])}
            for col in self._group_columns
        }

    @property
    def n_valid_bins(self) -> int:
        """Number of bins with valid fit results."""
        return int(np.sum(self._valid_mask))

    @property
    def sparsity(self) -> float:
        """Fraction of bins without valid fits."""
        total = self._valid_mask.size
        if total == 0:
            return 1.0
        return 1.0 - self.n_valid_bins / total

    def valid_mask(self) -> np.ndarray:
        """Boolean N-D array: True where fit is valid."""
        return self._valid_mask.copy()

    # ------------------------------------------------------------------ #
    #  Coordinate lookup
    # ------------------------------------------------------------------ #

    def _find_cell(
        self,
        positions: Dict[str, Union[float, np.ndarray]],
        bounds: str = 'clamp',
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Optional[np.ndarray]]:
        """
        Find the grid cell containing given position(s).

        Parameters
        ----------
        positions : dict[str, float or array]
            Coordinates per dimension.
        bounds : str
            'clamp' — clip to grid edges (default)
            'nan' — return NaN for out-of-range
            'extrapolate' — linear extrapolation from edge cells

        Returns
        -------
        lower_indices : list of int arrays (one per dimension)
        fractions : list of float arrays (one per dimension, [0,1] range)
        out_of_bounds : bool array or None
            True where any dimension is outside grid. None if bounds='clamp'.
        """
        D = len(self._group_columns)
        # Determine if scalar or vectorised
        sample = positions[self._group_columns[0]]
        is_scalar = np.isscalar(sample)
        n_points = 1 if is_scalar else len(sample)

        lower_indices = []
        fractions = []
        out_of_bounds = np.zeros(n_points, dtype=bool) if bounds != 'clamp' else None

        for d, col in enumerate(self._group_columns):
            centers = self._bin_centers[col]
            n_bins = len(centers)
            x = np.atleast_1d(np.asarray(positions[col], dtype=np.float64))

            # Find insertion point: idx such that centers[idx] <= x < centers[idx+1]
            idx = np.searchsorted(centers, x) - 1

            # Compute fractions before clipping for bounds detection
            if bounds != 'clamp':
                oob_low = x < centers[0]
                oob_high = x > centers[-1]
                out_of_bounds |= (oob_low | oob_high)

            if n_bins == 1:
                # Single-bin dimension: always use index 0, fraction 0
                idx = np.zeros_like(idx)
                f = np.zeros(n_points, dtype=np.float64)
                lower_indices.append(idx)
                fractions.append(f)
                continue

            if bounds == 'clamp':
                idx = np.clip(idx, 0, n_bins - 2)
            elif bounds == 'extrapolate':
                idx = np.clip(idx, 0, n_bins - 2)
            else:
                # bounds == 'nan': still clip indices for computation,
                # but mark out-of-bounds for later NaN filling
                idx = np.clip(idx, 0, n_bins - 2)

            # Fractional position within cell
            denom = centers[idx + 1] - centers[idx]
            denom = np.where(denom == 0, 1.0, denom)
            f = (x - centers[idx]) / denom

            if bounds == 'clamp':
                f = np.clip(f, 0.0, 1.0)
            # For extrapolate: f can be < 0 or > 1 (extrapolation)
            # For nan: f can be anything, will be NaN'd later

            lower_indices.append(idx)
            fractions.append(f)

        return lower_indices, fractions, out_of_bounds

    # ------------------------------------------------------------------ #
    #  Core evaluation
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        positions: Union[Dict[str, float], 'pd.DataFrame'],
        predictors: Union[Dict[str, float], 'pd.DataFrame'],
        method: str = 'multilinear',
        use_errors: bool = False,
        invalid_strategy: str = 'nan',
        bounds: str = 'clamp',
        targets: Optional[List[str]] = None,
    ) -> Union[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Evaluate the regression model at arbitrary positions.

        Parameters
        ----------
        positions : dict[str, float] or DataFrame
            Coordinates per dimension. Scalar for single point,
            array/DataFrame for batch evaluation.
        predictors : dict[str, float] or DataFrame
            Predictor values.
        method : str
            'nearest' or 'multilinear'.
        use_errors : bool
            If True, use inverse-variance weighting for interpolation.
        invalid_strategy : str
            'nan' — return NaN if any interpolation corner is invalid.
            'skip' — renormalise using valid corners only.
            'nearest_valid' — fall back to nearest valid bin.
        bounds : str
            'clamp' — clip to grid edges (default).
            'nan' — return NaN for out-of-range positions.
            'extrapolate' — linear extrapolation from edge cells.
        targets : list of str, optional
            Subset of targets to evaluate. None = all.

        Returns
        -------
        result : dict[str, float or np.ndarray]
            {target: predicted_value(s)} for each target.
        """
        eval_targets = targets if targets is not None else self._targets

        # Get interpolated coefficients
        coeffs = self.get_coefficients(
            positions, method=method, use_errors=use_errors,
            invalid_strategy=invalid_strategy, bounds=bounds,
            targets=eval_targets,
        )

        result = {}
        for tgt in eval_targets:
            tgt_coeffs = coeffs[tgt]
            # y = intercept + sum(slope_i * predictor_i)
            val = tgt_coeffs['intercept'].copy()
            for pred in self._predictor_columns:
                slope = tgt_coeffs[f'slope_{pred}']
                if HAS_PANDAS and isinstance(predictors, pd.DataFrame):
                    pred_val = predictors[pred].values
                elif isinstance(predictors, dict):
                    pred_val = np.atleast_1d(
                        np.asarray(predictors[pred], dtype=np.float64))
                else:
                    raise TypeError(
                        f"predictors must be dict or DataFrame, "
                        f"got {type(predictors)}")
                val = val + slope * pred_val
            # Squeeze scalar
            if val.size == 1:
                val = float(val.ravel()[0])
            result[tgt] = val

        return result

    @property
    def fit_model(self) -> str:
        """Fit model type: 'linear', 'gaussian', '<callable>', etc."""
        return self._fit_model

    @property
    def param_names(self) -> List[str]:
        """Parameter names for non-linear models (empty for linear)."""
        return list(self._param_names)

    def evaluate_model(
        self,
        positions: Union[Dict[str, float], 'pd.DataFrame'],
        x_query: Union[float, np.ndarray],
        model_func: Optional[Callable] = None,
        method: str = 'multilinear',
        use_errors: bool = False,
        invalid_strategy: str = 'nan',
        bounds: str = 'clamp',
        targets: Optional[List[str]] = None,
    ) -> Union[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Evaluate a non-linear model at arbitrary positions (Option A).

        Interpolates fitted parameters across the grid, then evaluates
        the model function with the interpolated parameters.

        Parameters
        ----------
        positions : dict[str, float] or DataFrame
            Coordinates per dimension (bin positions).
        x_query : float or array-like
            Predictor value(s) at which to evaluate the model.
            For named models, must be a scalar or 1-D array.
        model_func : callable, optional
            Model function ``f(x, *params) → y``, scipy.optimize.curve_fit
            compatible.  For named models, auto-resolved from the registry.
            Required for custom callables.
        method : str
            Interpolation method: ``'nearest'`` or ``'multilinear'``.
        use_errors : bool
            If True, use inverse-variance weighting for interpolation.
        invalid_strategy : str
            How to handle invalid bins: ``'nan'``, ``'skip'``, ``'nearest_valid'``.
        bounds : str
            Out-of-bounds handling: ``'clamp'``, ``'nan'``, ``'extrapolate'``.
        targets : list of str, optional
            Subset of targets to evaluate. None = all.

        Returns
        -------
        result : dict[str, float or np.ndarray]
            {target: predicted_value(s)} for each target.

        Raises
        ------
        ValueError
            If model_func is None and fit_model is not a registered named model.
        """
        if not self._is_nonlinear:
            raise ValueError(
                "evaluate_model() is for non-linear fits. "
                "Use evaluate() for linear fits."
            )

        # Resolve model function
        if model_func is None:
            model_func = self._resolve_model_func()

        eval_targets = targets if targets is not None else self._targets

        # Get interpolated parameters (reuses get_coefficients)
        coeffs = self.get_coefficients(
            positions, method=method, use_errors=use_errors,
            invalid_strategy=invalid_strategy, bounds=bounds,
            targets=eval_targets,
        )

        x = np.atleast_1d(np.asarray(x_query, dtype=np.float64))

        result = {}
        for tgt in eval_targets:
            tgt_coeffs = coeffs[tgt]
            # Extract parameter arrays (exclude _err and diagnostic keys)
            param_values = []
            for pn in self._param_names:
                if pn in tgt_coeffs:
                    param_values.append(tgt_coeffs[pn])
                else:
                    raise KeyError(
                        f"Parameter '{pn}' not found in interpolated "
                        f"coefficients for target '{tgt}'. "
                        f"Available: {list(tgt_coeffs.keys())}")

            # Evaluate: model_func(x, *params) → y
            # param_values are scalar or 1-D arrays from interpolation
            # x can be scalar or array
            try:
                params = [np.atleast_1d(p) for p in param_values]
                val = model_func(x, *[p for p in params])
            except Exception as exc:
                val = np.full_like(x, np.nan)

            # Squeeze scalar
            if val.size == 1:
                val = float(val.ravel()[0])
            result[tgt] = val

        return result

    def _resolve_model_func(self) -> Callable:
        """Auto-resolve model function from registry for named models."""
        try:
            from .groupby_regression_models import get_model
        except ImportError:
            from groupby_regression_models import get_model

        if self._fit_model in ('<callable>', 'linear'):
            raise ValueError(
                f"Cannot auto-resolve model function for fit_model="
                f"'{self._fit_model}'. Pass model_func= explicitly.")
        try:
            spec = get_model(self._fit_model)
            return spec.func
        except KeyError:
            raise ValueError(
                f"Model '{self._fit_model}' not in registry. "
                f"Pass model_func= explicitly or register the model."
            )

    def evaluate_params(
        self,
        positions: Union[Dict[str, float], 'pd.DataFrame'],
        method: str = 'multilinear',
        use_errors: bool = False,
        invalid_strategy: str = 'nan',
        bounds: str = 'clamp',
        targets: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Get interpolated non-linear parameters at arbitrary positions.

        Convenience wrapper around get_coefficients() for non-linear fits.
        Returns only the model parameters (excluding _err and diagnostic keys).

        Parameters
        ----------
        positions : dict[str, float] or DataFrame
            Coordinates per dimension.
        method, use_errors, invalid_strategy, bounds, targets :
            Same as get_coefficients().

        Returns
        -------
        params : dict[str, dict[str, float or np.ndarray]]
            {target: {param_name: interpolated_value}} for model parameters only.
        """
        coeffs = self.get_coefficients(
            positions, method=method, use_errors=use_errors,
            invalid_strategy=invalid_strategy, bounds=bounds,
            targets=targets,
        )
        eval_targets = targets if targets is not None else self._targets
        result = {}
        for tgt in eval_targets:
            tgt_params = {}
            for pn in self._param_names:
                if pn in coeffs[tgt]:
                    tgt_params[pn] = coeffs[tgt][pn]
            result[tgt] = tgt_params
        return result

    def evaluate_function(
        self,
        positions: Union[Dict[str, float], 'pd.DataFrame'],
        x_query: Union[float, np.ndarray],
        model_func: Optional[Callable] = None,
        method: str = 'multilinear',
        invalid_strategy: str = 'nan',
        bounds: str = 'clamp',
        targets: Optional[List[str]] = None,
    ) -> Union[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Evaluate a non-linear model at arbitrary positions (Option B).

        Evaluates the model function at the 2^D enclosing grid corners
        using each corner's fitted parameters, then interpolates the
        function values.  This respects parameter correlations because
        each corner evaluation uses a consistent parameter set.

        Parameters
        ----------
        positions : dict[str, float] or DataFrame
            Coordinates per dimension (bin positions).
        x_query : float or array-like
            Predictor value(s) at which to evaluate the model.
        model_func : callable, optional
            Model function ``f(x, *params) → y``.  Auto-resolved for
            named models.  Required for custom callables.
        method : str
            ``'multilinear'`` or ``'nearest'``.
        invalid_strategy : str
            ``'nan'``, ``'skip'``, or ``'nearest_valid'``.
        bounds : str
            ``'clamp'``, ``'nan'``, or ``'extrapolate'``.
        targets : list of str, optional
            Subset of targets.  None = all.

        Returns
        -------
        result : dict[str, float or np.ndarray]
            {target: predicted_value(s)} for each target.

        Notes
        -----
        Complexity is O(2^D × n_points × len(x_query)).  For typical
        TPC grids (D ≤ 4), this is 16 model evaluations per query point
        — negligible compared to I/O.

        This method is preferred over ``evaluate_model`` when fitted
        parameters are correlated (e.g., amplitude and sigma in Gaussian
        fits), as it avoids unphysical parameter combinations from
        independent interpolation.

        For C++/WASM portability, this method decomposes into:
        (1) grid cell lookup, (2) corner parameter extraction,
        (3) model evaluation at corners, (4) weighted average.
        Each step is a simple array operation.
        """
        if not self._is_nonlinear:
            raise ValueError(
                "evaluate_function() is for non-linear fits. "
                "Use evaluate() for linear fits."
            )

        if model_func is None:
            model_func = self._resolve_model_func()

        eval_targets = targets if targets is not None else self._targets

        # Convert positions to dict
        if HAS_PANDAS and isinstance(positions, pd.DataFrame):
            pos_dict = {col: positions[col].values
                        for col in self._group_columns}
            n_points = len(positions)
        elif isinstance(positions, dict):
            pos_dict = positions
            sample = positions[self._group_columns[0]]
            n_points = 1 if np.isscalar(sample) else len(sample)
        else:
            raise TypeError(
                f"positions must be dict or DataFrame, got {type(positions)}")

        x = np.atleast_1d(np.asarray(x_query, dtype=np.float64))

        if method == 'nearest':
            return self._eval_function_nearest(
                pos_dict, n_points, x, eval_targets, model_func, bounds)

        # --- Multilinear: evaluate model at 2^D corners, interpolate values ---
        lower_indices, fracs, out_of_bounds = self._find_cell(pos_dict, bounds)
        D = len(self._group_columns)

        corners = list(itertools.product([0, 1], repeat=D))
        n_corners = len(corners)

        # Build corner indices and geometric weights
        corner_indices = []
        geometric_weights = np.ones((n_corners, n_points), dtype=np.float64)

        for ci, corner in enumerate(corners):
            idx_tuple = []
            for d in range(D):
                idx_d = lower_indices[d] + corner[d]
                idx_d = np.clip(idx_d, 0, self._grid_shape[d] - 1)
                idx_tuple.append(idx_d)
                if corner[d] == 1:
                    geometric_weights[ci] *= fracs[d]
                else:
                    geometric_weights[ci] *= (1.0 - fracs[d])
            corner_indices.append(tuple(idx_tuple))

        # Corner validity
        corner_valid = np.ones((n_corners, n_points), dtype=bool)
        for ci, idx_tuple in enumerate(corner_indices):
            corner_valid[ci] = self._valid_mask[idx_tuple]

        # For each target: evaluate model at each corner, then interpolate
        result = {}
        for tgt in eval_targets:
            # Extract parameter arrays at each corner
            # corner_params[ci] = list of param arrays, each shape (n_points,)
            corner_params = []
            for ci, idx_tuple in enumerate(corner_indices):
                params_at_corner = []
                for pn in self._param_names:
                    if pn in self._coefficients[tgt]:
                        params_at_corner.append(
                            self._coefficients[tgt][pn][idx_tuple])
                    else:
                        params_at_corner.append(
                            np.full(n_points, np.nan))
                corner_params.append(params_at_corner)

            # Evaluate model at each corner for the given x_query
            # corner_fvals[ci] shape: (n_points,) if x is scalar,
            #                         (n_points, len(x)) if x is array
            is_scalar_x = (x.size == 1)

            if is_scalar_x:
                # Scalar x: one function value per corner per point
                corner_fvals = np.empty((n_corners, n_points),
                                        dtype=np.float64)
                for ci in range(n_corners):
                    for pt in range(n_points):
                        if not corner_valid[ci, pt]:
                            corner_fvals[ci, pt] = np.nan
                            continue
                        try:
                            p = [corner_params[ci][k][pt]
                                 for k in range(len(self._param_names))]
                            corner_fvals[ci, pt] = model_func(x[0], *p)
                        except Exception:
                            corner_fvals[ci, pt] = np.nan

                # Weighted average of function values
                vals = self._weighted_corner_average(
                    corner_fvals, geometric_weights, corner_valid,
                    n_points, n_corners, invalid_strategy, out_of_bounds)
                if vals.size == 1:
                    vals = float(vals.ravel()[0])
                result[tgt] = vals
            else:
                # Array x: evaluate model curve at each corner
                # Result shape: (n_points, len(x))
                n_x = len(x)
                corner_fvals = np.empty((n_corners, n_points, n_x),
                                        dtype=np.float64)
                for ci in range(n_corners):
                    for pt in range(n_points):
                        if not corner_valid[ci, pt]:
                            corner_fvals[ci, pt, :] = np.nan
                            continue
                        try:
                            p = [corner_params[ci][k][pt]
                                 for k in range(len(self._param_names))]
                            corner_fvals[ci, pt, :] = model_func(x, *p)
                        except Exception:
                            corner_fvals[ci, pt, :] = np.nan

                # Interpolate per x-value
                vals = np.empty((n_points, n_x), dtype=np.float64)
                for xi in range(n_x):
                    vals[:, xi] = self._weighted_corner_average(
                        corner_fvals[:, :, xi],
                        geometric_weights, corner_valid,
                        n_points, n_corners, invalid_strategy, out_of_bounds)

                # Squeeze
                if n_points == 1:
                    vals = vals.ravel()
                result[tgt] = vals

        return result

    def _eval_function_nearest(
        self,
        pos_dict: Dict[str, Any],
        n_points: int,
        x: np.ndarray,
        targets: List[str],
        model_func: Callable,
        bounds: str,
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Nearest-neighbor version of evaluate_function."""
        lower_indices, fracs, out_of_bounds = self._find_cell(pos_dict, bounds)
        D = len(self._group_columns)

        nearest_idx = []
        for d in range(D):
            idx = lower_indices[d].copy()
            upper = fracs[d] >= 0.5
            idx[upper] += 1
            idx = np.clip(idx, 0, self._grid_shape[d] - 1)
            nearest_idx.append(idx)
        idx_tuple = tuple(nearest_idx)

        result = {}
        for tgt in targets:
            params = []
            for pn in self._param_names:
                if pn in self._coefficients[tgt]:
                    params.append(self._coefficients[tgt][pn][idx_tuple])
                else:
                    params.append(np.full(n_points, np.nan))

            is_scalar_x = (x.size == 1)
            if is_scalar_x:
                vals = np.empty(n_points, dtype=np.float64)
                for pt in range(n_points):
                    try:
                        p = [params[k][pt]
                             for k in range(len(self._param_names))]
                        vals[pt] = model_func(x[0], *p)
                    except Exception:
                        vals[pt] = np.nan
            else:
                vals = np.empty((n_points, len(x)), dtype=np.float64)
                for pt in range(n_points):
                    try:
                        p = [params[k][pt]
                             for k in range(len(self._param_names))]
                        vals[pt, :] = model_func(x, *p)
                    except Exception:
                        vals[pt, :] = np.nan
                if n_points == 1:
                    vals = vals.ravel()

            if out_of_bounds is not None:
                if vals.ndim == 1:
                    vals[out_of_bounds] = np.nan
                else:
                    vals[out_of_bounds, :] = np.nan

            if vals.size == 1:
                vals = float(vals.ravel()[0])
            result[tgt] = vals

        return result

    @staticmethod
    def _weighted_corner_average(
        corner_vals: np.ndarray,
        geometric_weights: np.ndarray,
        corner_valid: np.ndarray,
        n_points: int,
        n_corners: int,
        invalid_strategy: str,
        out_of_bounds: Optional[np.ndarray],
    ) -> np.ndarray:
        """Weighted average of corner values with validity handling.

        Parameters
        ----------
        corner_vals : (n_corners, n_points)
        geometric_weights : (n_corners, n_points)
        corner_valid : (n_corners, n_points)
        """
        weights = geometric_weights.copy()

        if invalid_strategy == 'nan':
            any_invalid = np.zeros(n_points, dtype=bool)
            for ci in range(n_corners):
                has_weight = geometric_weights[ci] > 1e-15
                any_invalid |= (has_weight & ~corner_valid[ci])
        elif invalid_strategy in ('skip', 'nearest_valid'):
            weights = np.where(corner_valid, weights, 0.0)
            corner_vals = np.where(corner_valid, corner_vals, 0.0)
        else:
            raise ValueError(f"Unknown invalid_strategy '{invalid_strategy}'")

        total_weight = np.sum(weights, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            vals = np.where(
                total_weight > 0,
                np.sum(weights * corner_vals, axis=0) / total_weight,
                np.nan)

        if invalid_strategy == 'nan':
            vals[any_invalid] = np.nan

        if out_of_bounds is not None:
            vals[out_of_bounds] = np.nan

        return vals

    def get_coefficients(
        self,
        positions: Union[Dict[str, float], 'pd.DataFrame'],
        method: str = 'multilinear',
        use_errors: bool = False,
        invalid_strategy: str = 'nan',
        bounds: str = 'clamp',
        targets: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Get interpolated coefficients (and errors) at arbitrary positions.

        Returns
        -------
        coeffs : dict[str, dict[str, float or np.ndarray]]
            {target: {coeff_name: value(s)}} for each target.
        """
        eval_targets = targets if targets is not None else self._targets

        # Convert DataFrame to dict
        if HAS_PANDAS and isinstance(positions, pd.DataFrame):
            pos_dict = {col: positions[col].values for col in self._group_columns}
            n_points = len(positions)
        elif isinstance(positions, dict):
            pos_dict = positions
            sample = positions[self._group_columns[0]]
            n_points = 1 if np.isscalar(sample) else len(sample)
        else:
            raise TypeError(
                f"positions must be dict or DataFrame, got {type(positions)}")

        # Find cells
        lower_indices, fracs, out_of_bounds = self._find_cell(pos_dict, bounds)

        if method == 'nearest':
            return self._eval_nearest(
                lower_indices, fracs, n_points, eval_targets, out_of_bounds)
        elif method == 'multilinear':
            return self._eval_multilinear(
                lower_indices, fracs, n_points, eval_targets,
                use_errors, invalid_strategy, out_of_bounds)
        else:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Use 'nearest' or 'multilinear'.")

    def _eval_nearest(
        self,
        lower_indices: List[np.ndarray],
        fracs: List[np.ndarray],
        n_points: int,
        targets: List[str],
        out_of_bounds: Optional[np.ndarray],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Nearest-neighbor evaluation (snap to closest bin center)."""
        D = len(self._group_columns)

        # Round to nearest bin
        nearest_idx = []
        for d in range(D):
            # If fraction >= 0.5, use upper bin; else lower bin
            idx = lower_indices[d].copy()
            upper = fracs[d] >= 0.5
            idx[upper] += 1
            # Clip to valid range
            idx = np.clip(idx, 0, self._grid_shape[d] - 1)
            nearest_idx.append(idx)

        # Convert to tuple for indexing
        idx_tuple = tuple(nearest_idx)

        result = {}
        for tgt in targets:
            tgt_result = {}
            for key, grid in self._coefficients[tgt].items():
                vals = grid[idx_tuple].copy()
                if out_of_bounds is not None:
                    vals[out_of_bounds] = np.nan
                tgt_result[key] = vals
            result[tgt] = tgt_result

        return result

    def _eval_multilinear(
        self,
        lower_indices: List[np.ndarray],
        fracs: List[np.ndarray],
        n_points: int,
        targets: List[str],
        use_errors: bool,
        invalid_strategy: str,
        out_of_bounds: Optional[np.ndarray],
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        N-D multilinear interpolation (vectorised over points).

        For each point, computes weighted average over 2^D corners.
        """
        D = len(self._group_columns)

        # Pre-compute all 2^D corners
        corners = list(itertools.product([0, 1], repeat=D))
        n_corners = len(corners)

        # Build corner indices and geometric weights for all points
        # Shape: (n_corners, n_points) per dimension
        corner_indices = []
        geometric_weights = np.ones((n_corners, n_points), dtype=np.float64)

        for ci, corner in enumerate(corners):
            idx_tuple = []
            for d in range(D):
                idx_d = lower_indices[d] + corner[d]
                # Clip to valid range
                idx_d = np.clip(idx_d, 0, self._grid_shape[d] - 1)
                idx_tuple.append(idx_d)

                # Weight: f for upper corner, (1-f) for lower
                if corner[d] == 1:
                    geometric_weights[ci] *= fracs[d]
                else:
                    geometric_weights[ci] *= (1.0 - fracs[d])

            corner_indices.append(tuple(idx_tuple))

        # Check which corners are valid
        corner_valid = np.ones((n_corners, n_points), dtype=bool)
        for ci, idx_tuple in enumerate(corner_indices):
            corner_valid[ci] = self._valid_mask[idx_tuple]
            # Bounds check: corners outside grid
            for d in range(D):
                actual_idx = lower_indices[d] + corners[ci][d]
                corner_valid[ci] &= (actual_idx >= 0)
                corner_valid[ci] &= (actual_idx < self._grid_shape[d])
                # For single-bin dimensions, upper corner == lower corner
                # (already handled by clip in corner_indices), so valid

        # Interpolate each target's coefficients
        result = {}
        for tgt in targets:
            tgt_result = {}
            for key, grid in self._coefficients[tgt].items():
                # Get corner values: (n_corners, n_points)
                corner_vals = np.empty((n_corners, n_points), dtype=np.float64)
                for ci, idx_tuple in enumerate(corner_indices):
                    corner_vals[ci] = grid[idx_tuple]

                # Apply weights
                if use_errors and (key in ('intercept',) or (
                        key.startswith('slope_') and not key.endswith('_err'))):
                    # Use inverse-variance weighting from error columns
                    err_key = key + '_err'
                    if err_key in self._coefficients[tgt]:
                        err_grid = self._coefficients[tgt][err_key]
                        corner_errs = np.empty((n_corners, n_points),
                                               dtype=np.float64)
                        for ci, idx_tuple in enumerate(corner_indices):
                            corner_errs[ci] = err_grid[idx_tuple]
                        # Inverse variance: w = 1/err^2
                        with np.errstate(divide='ignore', invalid='ignore'):
                            ivar_weights = np.where(
                                corner_errs > 0,
                                1.0 / (corner_errs ** 2),
                                0.0)
                        weights = geometric_weights * ivar_weights
                    else:
                        weights = geometric_weights.copy()
                else:
                    weights = geometric_weights.copy()

                # Apply invalid_strategy
                if invalid_strategy == 'nan':
                    # Any invalid corner → entire point is NaN
                    any_invalid = np.zeros(n_points, dtype=bool)
                    for ci in range(n_corners):
                        # Only check corners with non-zero weight
                        has_weight = geometric_weights[ci] > 1e-15
                        any_invalid |= (has_weight & ~corner_valid[ci])
                elif invalid_strategy == 'skip':
                    weights = np.where(corner_valid, weights, 0.0)
                    # Zero NaN corner values to prevent NaN propagation
                    corner_vals = np.where(corner_valid, corner_vals, 0.0)
                elif invalid_strategy == 'nearest_valid':
                    # For invalid corners, set weight to 0
                    # Will fall back to valid corners with renormalization
                    weights = np.where(corner_valid, weights, 0.0)
                    corner_vals = np.where(corner_valid, corner_vals, 0.0)
                else:
                    raise ValueError(
                        f"Unknown invalid_strategy '{invalid_strategy}'")

                # Weighted sum
                total_weight = np.sum(weights, axis=0)
                with np.errstate(divide='ignore', invalid='ignore'):
                    vals = np.where(
                        total_weight > 0,
                        np.sum(weights * corner_vals, axis=0) / total_weight,
                        np.nan)

                # Apply NaN for invalid_strategy='nan'
                if invalid_strategy == 'nan':
                    vals[any_invalid] = np.nan

                # Apply out-of-bounds NaN
                if out_of_bounds is not None:
                    vals[out_of_bounds] = np.nan

                tgt_result[key] = vals
            result[tgt] = tgt_result

        return result

    # ------------------------------------------------------------------ #
    #  Grid operations
    # ------------------------------------------------------------------ #

    def evaluate_grid(
        self,
        grid_coords: Dict[str, np.ndarray],
        predictors: Dict[str, float],
        method: str = 'multilinear',
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        """
        Evaluate on a regular output grid (for visualization).

        Parameters
        ----------
        grid_coords : dict[str, array-like]
            Output grid coordinates per dimension. Can be finer than
            input bins (upsampling).
        predictors : dict[str, float]
            Fixed predictor values.

        Returns
        -------
        result : dict[str, np.ndarray]
            {target: N-D array of predicted values on the output grid}
        """
        # Build meshgrid of positions
        arrays = [np.asarray(grid_coords[col]) for col in self._group_columns]
        mesh = np.meshgrid(*arrays, indexing='ij')
        flat_positions = {
            col: mesh[i].ravel()
            for i, col in enumerate(self._group_columns)
        }

        # Broadcast predictors
        n_points = mesh[0].size
        flat_predictors = {
            pred: np.full(n_points, val)
            for pred, val in predictors.items()
        }

        # Evaluate
        flat_result = self.evaluate(
            flat_positions, flat_predictors, method=method, **kwargs)

        # Reshape to grid
        out_shape = tuple(len(grid_coords[col]) for col in self._group_columns)
        result = {}
        for tgt, vals in flat_result.items():
            result[tgt] = np.asarray(vals).reshape(out_shape)
        return result

    def coefficient_grid(self, target: str, coeff_name: str) -> np.ndarray:
        """
        Extract a single coefficient as an N-D array.

        Parameters
        ----------
        target : str
            Target name, e.g. 'dX'.
        coeff_name : str
            e.g. 'intercept', 'slope_meanIDC', 'intercept_err'.

        Returns
        -------
        grid : np.ndarray
            N-D array with shape = grid_shape. NaN where fit is invalid.
        """
        if target not in self._coefficients:
            raise KeyError(f"Unknown target '{target}'. "
                           f"Available: {self._targets}")
        if coeff_name not in self._coefficients[target]:
            raise KeyError(
                f"Unknown coefficient '{coeff_name}' for target '{target}'. "
                f"Available: {list(self._coefficients[target].keys())}")
        return self._coefficients[target][coeff_name].copy()

    # ------------------------------------------------------------------ #
    #  Export / Import
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """
        Export to a JSON-serialisable dict.

        The exported dict uses flat (C-order) arrays for coefficients
        and valid_mask, suitable for JSON or binary serialization.
        """
        coefficients = {}
        for tgt in self._targets:
            tgt_dict = {}
            for key, arr in self._coefficients[tgt].items():
                tgt_dict[key] = {
                    'data': arr.ravel().tolist(),
                    'dtype': str(arr.dtype),
                }
            coefficients[tgt] = tgt_dict

        return {
            'schema_version': self.SCHEMA_VERSION,
            'group_columns': self._group_columns,
            'predictor_columns': self._predictor_columns,
            'targets': self._targets,
            'grid_shape': list(self._grid_shape),
            'bin_centers': {
                col: arr.tolist()
                for col, arr in self._bin_centers.items()
            },
            'coefficients': coefficients,
            'valid_mask': self._valid_mask.ravel().tolist(),
            'metadata': {
                'n_valid': self.n_valid_bins,
                'sparsity': round(self.sparsity, 4),
                **self._metadata,
            },
        }

    def to_json(self, path: str, compress: bool = True) -> None:
        """
        Export to JSON file (optionally gzipped).

        Parameters
        ----------
        path : str
            Output file path. If compress=True and path doesn't end
            with '.gz', '.gz' is appended.
        compress : bool
            Whether to gzip the output.
        """
        data = self.to_dict()
        json_str = json.dumps(data, separators=(',', ':'))

        if compress:
            if not path.endswith('.gz'):
                path = path + '.gz'
            with gzip.open(path, 'wt', encoding='utf-8') as f:
                f.write(json_str)
        else:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GroupByRegressionEvaluator':
        """
        Construct from exported dict (inverse of to_dict).

        Parameters
        ----------
        data : dict
            Dict as produced by to_dict().
        """
        schema_version = data.get('schema_version', '1.0')
        if schema_version != cls.SCHEMA_VERSION:
            warnings.warn(
                f"Schema version mismatch: file has {schema_version}, "
                f"evaluator expects {cls.SCHEMA_VERSION}")

        grid_shape = tuple(data['grid_shape'])
        group_columns = data['group_columns']
        predictor_columns = data['predictor_columns']
        targets = data['targets']

        bin_centers = {
            col: np.array(arr, dtype=np.float64)
            for col, arr in data['bin_centers'].items()
        }

        coefficients = {}
        for tgt in targets:
            tgt_dict = {}
            for key, spec in data['coefficients'][tgt].items():
                arr = np.array(spec['data'], dtype=np.float64)
                tgt_dict[key] = arr.reshape(grid_shape)
            coefficients[tgt] = tgt_dict

        valid_mask = np.array(data['valid_mask'], dtype=bool).reshape(grid_shape)

        metadata = data.get('metadata', {})
        # Remove computed fields
        metadata.pop('n_valid', None)
        metadata.pop('sparsity', None)

        return cls(
            grid_shape=grid_shape,
            group_columns=group_columns,
            predictor_columns=predictor_columns,
            targets=targets,
            bin_centers=bin_centers,
            coefficients=coefficients,
            valid_mask=valid_mask,
            metadata=metadata,
        )

    @classmethod
    def from_json(cls, path: str) -> 'GroupByRegressionEvaluator':
        """Load from JSON file (supports gzipped)."""
        if path.endswith('.gz'):
            with gzip.open(path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dfGB(
        cls,
        dfGB: 'pd.DataFrame',
        group_columns: Optional[List[str]] = None,
        predictor_columns: Optional[List[str]] = None,
        targets: Optional[Union[str, List[str]]] = None,
        bin_centers: Optional[Dict[str, np.ndarray]] = None,
        bin_edges: Optional[Dict[str, np.ndarray]] = None,
        suffix: str = '',
        metadata: Optional[Dict[str, Any]] = None,
    ) -> 'GroupByRegressionEvaluator':
        """
        Construct from a GroupBy regression output DataFrame.

        This is the pandas-specific convenience constructor. It parses
        column names, strips suffixes, and builds N-D arrays.

        The recommended usage is to pass the ``metadata`` dict returned by
        ``make_parallel_fit_v4(..., return_metadata=True)`` or
        ``make_sliding_window_fit(..., return_metadata=True)``.
        The metadata contains ``gb_columns``, ``fit_columns`` (targets),
        ``linear_columns`` (predictors), ``suffix``, and ``fit_intercept``,
        so individual parameters can be omitted.

        Parameters
        ----------
        dfGB : pd.DataFrame
            Regression output (from make_parallel_fit_v4,
            make_sliding_window_fit, etc.)
        group_columns : list of str, optional
            Bin dimension column names. Extracted from metadata if provided.
        predictor_columns : list of str, optional
            Predictor names. Extracted from metadata if provided.
        targets : str or list of str, optional
            Target variable name(s). Extracted from metadata if provided.
        bin_centers : dict[str, array-like], optional
            Physical coordinates per dimension. If None, unique values
            from dfGB[group_columns] are used as float coordinates.
        bin_edges : dict[str, array-like], optional
            Bin edges per dimension. Centers computed as midpoints.
            Mutually exclusive with bin_centers.
        suffix : str
            Column suffix to strip (e.g., '_sw', '_fit').
            Extracted from metadata if provided.
        metadata : dict, optional
            Metadata dict from make_parallel_fit_v4 or make_sliding_window_fit.
            If provided, extracts group_columns, fit_columns (targets),
            linear_columns (predictors), suffix, and fit_intercept.
            Explicit parameters override metadata values.

        Examples
        --------
        >>> # Recommended: with metadata (no guessing)
        >>> df_out, dfGB, metadata = make_parallel_fit_v4(
        ...     ..., return_metadata=True)
        >>> ev = GroupByRegressionEvaluator.from_dfGB(dfGB, metadata=metadata)

        >>> # Without metadata: must specify all parameters
        >>> ev = GroupByRegressionEvaluator.from_dfGB(
        ...     dfGB, group_columns=['xBin', 'yBin'],
        ...     predictor_columns=['pred'], targets='value', suffix='_fit')
        """
        if not HAS_PANDAS:
            raise ImportError("pandas is required for from_dfGB")

        # Extract from metadata (explicit params override)
        if metadata is not None:
            columns_info = metadata.get('columns', {})
            params_info = metadata.get('parameters', {})
            if group_columns is None:
                group_columns = columns_info.get('gb_columns')
            if targets is None:
                targets = columns_info.get('fit_columns')
            if predictor_columns is None:
                predictor_columns = columns_info.get('linear_columns')
            if suffix == '' and 'suffix' in params_info:
                suffix = params_info['suffix']

        # Validate that required parameters are available
        if group_columns is None:
            raise ValueError(
                "group_columns must be specified (either directly or via metadata)")
        if targets is None:
            raise ValueError(
                "targets must be specified (either directly or via metadata). "
                "Use return_metadata=True in the regression function.")
        if predictor_columns is None:
            raise ValueError(
                "predictor_columns must be specified (either directly or via metadata). "
                "Use return_metadata=True in the regression function.")

        if isinstance(targets, str):
            targets = [targets]

        if bin_centers is not None and bin_edges is not None:
            raise ValueError(
                "bin_centers and bin_edges are mutually exclusive")

        # Step 1: Determine unique bin values and grid shape
        unique_vals = {}
        for col in group_columns:
            uv = np.sort(dfGB[col].unique())
            unique_vals[col] = uv

        grid_shape = tuple(len(unique_vals[col]) for col in group_columns)

        # Step 2: Determine bin centers
        if bin_centers is not None:
            bc = {col: np.asarray(bin_centers[col], dtype=np.float64)
                  for col in group_columns}
        elif bin_edges is not None:
            bc = {}
            for col in group_columns:
                edges = np.asarray(bin_edges[col], dtype=np.float64)
                bc[col] = 0.5 * (edges[:-1] + edges[1:])
        else:
            # Use unique values as float coordinates
            bc = {col: unique_vals[col].astype(np.float64)
                  for col in group_columns}

        # Step 3: Build index mapping (bin_value → array index)
        index_maps = {}
        for col in group_columns:
            val_to_idx = {v: i for i, v in enumerate(unique_vals[col])}
            index_maps[col] = val_to_idx

        # Step 4: Parse column names and fill coefficient arrays
        # Known coefficient patterns:
        #   {target}_intercept{suffix} → 'intercept'
        #   {target}_slope_{pred}{suffix} → 'slope_{pred}'
        #   {target}_intercept_err{suffix} → 'intercept_err'
        #   {target}_slope_{pred}_err{suffix} → 'slope_{pred}_err'
        #   {target}_rms{suffix} → 'rms'
        #   {target}_rmse{suffix} → 'rmse'
        #   {target}_r_squared{suffix} → 'r_squared'
        #   {target}_std{suffix} → 'std'
        #   {target}_median{suffix} → 'median'
        #   {target}_mad{suffix} → 'mad'
        #   {target}_n_fitted{suffix} → 'n_fitted'

        # Detect if this is a non-linear fit
        params_info = (metadata or {}).get('parameters', {})
        self_is_nonlinear = params_info.get('fit_model', 'linear') != 'linear'

        coefficients = {}
        for tgt in targets:
            tgt_coeffs = {}
            prefix = f'{tgt}_'

            # Find all columns matching this target
            for col_name in dfGB.columns:
                if not col_name.startswith(prefix):
                    continue
                # Strip prefix and suffix
                inner = col_name[len(prefix):]
                if suffix and inner.endswith(suffix):
                    inner = inner[:-len(suffix)]
                elif suffix:
                    # Column doesn't have expected suffix, skip
                    continue

                # Skip group columns and non-coefficient columns
                if inner in ('pull', 'pull_mad', 'prediction', 'residual'):
                    continue

                # Map to canonical key
                canonical = inner  # e.g., 'intercept', 'slope_spaceCharge', etc.
                tgt_coeffs[canonical] = inner  # placeholder for column name

            # Also check for nEffective (V4 style, no target prefix)
            # and quality_flag (SW style)
            # These are grid-level, not per-target — skip for now

            # Fill N-D arrays
            filled_coeffs = {}
            for canonical in list(tgt_coeffs.keys()):
                col_name = f'{tgt}_{canonical}{suffix}'
                if col_name not in dfGB.columns:
                    continue
                # Skip non-numeric columns (e.g., quality_flag)
                col_dtype = dfGB[col_name].dtype
                try:
                    is_numeric = np.issubdtype(col_dtype, np.number)
                except TypeError:
                    # Pandas extension dtypes (StringDtype, etc.)
                    is_numeric = False
                if not is_numeric:
                    continue
                grid = np.full(grid_shape, np.nan, dtype=np.float64)
                for _, row in dfGB.iterrows():
                    idx = tuple(
                        index_maps[col][row[col]]
                        for col in group_columns
                    )
                    grid[idx] = row[col_name]
                filled_coeffs[canonical] = grid

            # Ensure minimum required coefficients exist
            if self_is_nonlinear:
                # Non-linear: just need at least one coefficient grid
                if len(filled_coeffs) == 0:
                    raise ValueError(
                        f"No coefficient columns found for target '{tgt}' "
                        f"with suffix '{suffix}'. "
                        f"Available: {list(dfGB.columns)}")
            else:
                # Linear: require intercept + slope_*
                if 'intercept' not in filled_coeffs:
                    raise ValueError(
                        f"Column '{tgt}_intercept{suffix}' not found in dfGB. "
                        f"Available: {list(dfGB.columns)}")
                for pred in predictor_columns:
                    if f'slope_{pred}' not in filled_coeffs:
                        raise ValueError(
                            f"Column '{tgt}_slope_{pred}{suffix}' not found. "
                            f"Available: {list(dfGB.columns)}")

            coefficients[tgt] = filled_coeffs

        # Step 5: Build valid mask (from first coefficient of first target)
        first_tgt_coeffs = coefficients[targets[0]]
        first_key = next(iter(first_tgt_coeffs))
        valid_mask_arr = ~np.isnan(first_tgt_coeffs[first_key])

        return cls(
            grid_shape=grid_shape,
            group_columns=group_columns,
            predictor_columns=predictor_columns,
            targets=targets,
            bin_centers=bc,
            coefficients=coefficients,
            valid_mask=valid_mask_arr,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------ #
    #  String representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        model_info = f", fit_model='{self._fit_model}'" if self._is_nonlinear else ''
        return (
            f"GroupByRegressionEvaluator("
            f"targets={self._targets}, "
            f"grid_shape={self._grid_shape}, "
            f"dims={self._group_columns}, "
            f"predictors={self._predictor_columns}, "
            f"valid={self.n_valid_bins}/{self._valid_mask.size}"
            f"{model_info}"
            f")"
        )
