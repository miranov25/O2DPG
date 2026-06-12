"""Phase 13.42.DF — Inline fits for hist/profile/scatter.

Engine: scipy.optimize.curve_fit (architect directive Q-3).
Public API: normalize_fit_spec(), dispatch_fit(), register_fit(), available_fits().

Per architect direction (2026-05-22): inline fits are COMPLEMENTARY to the
existing GBregression + AliasDataFrame subframe + alias arithmetic pattern,
NOT a replacement. Use cases:
  - Visual fit on single curve when GBregression is overkill
  - Pull-distribution validation: fit gauss to (val-val_fit)/val_fit_mad
  - Compression QA: fit gauss to raw - decompressed residual
  - 1D-histogram + fit for legacy/ML-projection demographic

Initial-parameter resolution (4-tier, Phase 13.42 v1.2 §3.3):
  1. User-supplied 'initial'   (highest priority, always wins)
  2. User-supplied 'guess'     callable: (x, y) -> List[float]
  3. Registry-bound heuristic  (for predefined names)
  4. scipy default p0=[1]*N    (last resort, with UserWarning for user callables)

Composition (standard dfdraw vector convention per Phase 13.16):
  n_fits=1, any n_curves   → broadcast
  n_fits=n_curves          → per-channel pairing
  n_fits>1, n_curves=1     → compound (all fits overlaid on the curve)
  Otherwise                → ValueError (length mismatch)
"""

from typing import Union, Callable, Tuple, Optional, List, Dict, Any
import inspect
import warnings

import numpy as np
import scipy.optimize


# Module-level registry: name → callable f(x, *params)
_FIT_REGISTRY: Dict[str, Callable] = {}
# Per-registry-name initial-parameter heuristic: name → fn(x, y) → list of p0
_FIT_INITIAL_HEURISTICS: Dict[str, Callable] = {}


_VALID_DICT_KEYS = frozenset({
    'fun', 'initial', 'guess', 'range', 'bounds', 'use_errors',
    'show_params', 'label', 'raise_on_failure',
    'redchi_warn_threshold', 'method', 'kwargs',
    # Phase 13.57.DF DR-5 (architect OK 2026-06-12): 'p0' is a kept-forever
    # synonym of 'initial' — scipy.optimize.curve_fit's own name for the
    # starting values (PRINCIPLES P-5). 'guess' is NOT aliased: it is a
    # callable (x, y) -> p0 (a generator of starting values, fits.py tier
    # docs above), semantically distinct from a start vector.
    'p0',
})


# =============================================================================
# Registry API
# =============================================================================

def register_fit(name: str, callable_fn: Callable,
                 initial_heuristic: Optional[Callable] = None,
                 replace: bool = False) -> None:
    """Register a fit by name.

    Parameters
    ----------
    name : str
        Registry key (case-insensitive — stored as lowercase).
    callable_fn : callable
        scipy.curve_fit-compatible function ``f(x, *params)``.
    initial_heuristic : callable, optional
        Function ``(x, y) -> list of p0`` that produces an initial guess
        from the post-range/post-NaN-filtered data. Used when the user
        does NOT supply ``'initial'`` or ``'guess'`` in the fit dict.
    replace : bool, default False
        If True, overwrite an existing registry entry with the same name.

    Raises
    ------
    ValueError
        If ``name`` already exists and ``replace=False``.

    Notes
    -----
    Aliased names (e.g. 'gauss' and 'gaussian') are independent registry
    entries; ``replace=True`` updates only the named entry, not aliases.
    """
    key = name.lower()
    if key in _FIT_REGISTRY and not replace:
        raise ValueError(
            f"[dfdraw.fits] Fit name {name!r} already registered. "
            f"Pass replace=True to overwrite, or pick a different name."
        )
    _FIT_REGISTRY[key] = callable_fn
    if initial_heuristic is not None:
        _FIT_INITIAL_HEURISTICS[key] = initial_heuristic


def available_fits() -> List[str]:
    """Return sorted list of registered fit names."""
    return sorted(_FIT_REGISTRY.keys())


# =============================================================================
# Predefined lineshapes + initial-parameter heuristics
# =============================================================================

def _gaussian(x, amplitude, center, sigma):
    """Gaussian: A * exp(-(x-c)^2 / (2 sigma^2))."""
    return amplitude * np.exp(-(x - center) ** 2 / (2.0 * sigma ** 2))


def _gaussian_guess(x, y):
    """Heuristic for predefined gauss (Phase 13.42 v1.3 CP2-1):

      amplitude = y.max()
      center    = x at argmax(y)
      sigma     = weighted std of x using y as weights — the empirical
                  distribution sigma, NOT the x-axis range.

    The earlier (v1.2) heuristic used ``np.nanstd(x)/2`` which is the
    x-axis spread — wrong when (e.g.) hist range is [-100, 100] but the
    peak's true sigma is 0.5. The weighted-std form is bounded by the
    actual distribution width and converges robustly for both wide and
    narrow peaks on arbitrary x-axes.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = float(np.nanmax(y))
    c = float(x[np.nanargmax(y)])
    # Weighted std: sqrt(sum(y_i * (x_i - c)^2) / sum(y_i))
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if int(valid.sum()) > 1:
        xv, yv = x[valid], y[valid]
        wsum = float(np.sum(yv))
        if wsum > 0:
            s = float(np.sqrt(np.sum(yv * (xv - c) ** 2) / wsum))
            if s > 0 and np.isfinite(s):
                return [a, c, s]
    # Fallback when weighted-std path fails (sparse/empty data)
    fallback_s = float(np.nanstd(x)) / 4.0
    if not np.isfinite(fallback_s) or fallback_s <= 0:
        fallback_s = 1.0
    return [a, c, fallback_s]


def _linear(x, slope, intercept):
    return slope * x + intercept


def _linear_guess(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 2:
        return [0.0, float(np.nanmean(y)) if np.any(np.isfinite(y)) else 0.0]
    return np.polyfit(x[m], y[m], 1).tolist()


def _polynomial_factory(degree):
    def _poly(x, *coeffs):
        # Ascending powers: coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + ...
        return np.polynomial.polynomial.polyval(np.asarray(x, dtype=float),
                                                np.asarray(coeffs, dtype=float))
    _poly.__name__ = f'_polynomial_deg{degree}'
    return _poly


def _polynomial_guess_factory(degree):
    def _guess(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < (degree + 1):
            # Not enough points — return a flat guess
            mean = float(np.nanmean(y)) if np.any(np.isfinite(y)) else 0.0
            return [mean] + [0.0] * degree
        # np.polyfit returns highest-degree-first; reverse to ascending order
        return list(np.polyfit(x[m], y[m], degree)[::-1])
    return _guess


def _exponential(x, amplitude, decay):
    """Exponential: A * exp(-decay * x)."""
    return amplitude * np.exp(-decay * np.asarray(x, dtype=float))


def _exponential_guess(x, y):
    """Heuristic: A = y.max(); decay from log-linear fit on positive y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = float(np.nanmax(y))
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if int(valid.sum()) >= 2:
        try:
            # log(y) = log(A) - decay*x  →  slope = -decay
            slope, _intercept = np.polyfit(x[valid], np.log(y[valid]), 1)
            decay = float(-slope)
            if np.isfinite(decay) and decay != 0:
                return [a, decay]
        except (np.linalg.LinAlgError, ValueError):
            pass
    return [a, 1.0]


def _lorentzian(x, amplitude, center, gamma):
    """Lorentzian (Cauchy): A * gamma^2 / ((x-c)^2 + gamma^2)."""
    x = np.asarray(x, dtype=float)
    return amplitude * gamma ** 2 / ((x - center) ** 2 + gamma ** 2)


def _lorentzian_guess(x, y):
    """Heuristic: same shape parameters as gauss; gamma ≈ HWHM."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    a = float(np.nanmax(y))
    c = float(x[np.nanargmax(y)])
    # gamma is HWHM; approximate from weighted std (close enough as p0)
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if int(valid.sum()) > 1:
        xv, yv = x[valid], y[valid]
        wsum = float(np.sum(yv))
        if wsum > 0:
            s = float(np.sqrt(np.sum(yv * (xv - c) ** 2) / wsum))
            if s > 0 and np.isfinite(s):
                return [a, c, s]
    fallback = float(np.nanstd(x)) / 4.0 or 1.0
    return [a, c, fallback]


def _powerlaw(x, amplitude, exponent):
    """Power-law: A * x^exponent (x must be positive)."""
    x = np.asarray(x, dtype=float)
    # Guard against negative/zero x — caller is expected to apply 'range'.
    safe = np.where(x > 0, x, np.nan)
    return amplitude * np.power(safe, exponent)


def _powerlaw_guess(x, y):
    """Heuristic: log-log linear fit on positive (x, y) pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(valid.sum()) >= 2:
        try:
            # log(y) = log(A) + exponent * log(x)
            slope, intercept = np.polyfit(np.log(x[valid]), np.log(y[valid]), 1)
            a = float(np.exp(intercept))
            exponent = float(slope)
            if np.isfinite(a) and np.isfinite(exponent):
                return [a, exponent]
        except (np.linalg.LinAlgError, ValueError):
            pass
    # Fallback
    return [float(np.nanmax(y)) if np.any(np.isfinite(y)) else 1.0, 1.0]


def _init_registry():
    """Populate the predefined registry (called once at module import)."""
    register_fit('gauss',       _gaussian,    _gaussian_guess)
    register_fit('gaussian',    _gaussian,    _gaussian_guess)
    register_fit('gaus',        _gaussian,    _gaussian_guess)  # Phase 13.46 C-1: ROOT TF1 convention
    register_fit('linear',      _linear,      _linear_guess)
    register_fit('pol1',        _linear,      _linear_guess)
    register_fit(
        'pol0',
        lambda x, c: np.full_like(np.asarray(x, dtype=float), c, dtype=float),
        lambda x, y: [float(np.nanmean(y)) if np.any(np.isfinite(y)) else 0.0],
    )
    for deg in (2, 3, 4, 5):
        register_fit(f'pol{deg}',
                     _polynomial_factory(deg),
                     _polynomial_guess_factory(deg))
    register_fit('expo',        _exponential, _exponential_guess)
    register_fit('exponential', _exponential, _exponential_guess)
    register_fit('lorentz',     _lorentzian,  _lorentzian_guess)
    register_fit('lorentzian',  _lorentzian,  _lorentzian_guess)
    register_fit('powerlaw',    _powerlaw,    _powerlaw_guess)


_init_registry()


# =============================================================================
# Spec normalization
# =============================================================================

def normalize_fit_spec(fit, n_curves: int) -> List[List[Dict]]:
    """Normalize any of the accepted ``fit=`` forms to ``List[List[Dict]]``.

    Returns
    -------
    result : list of lists of dicts
        ``result[i]`` = list of fit-dicts to apply to curve ``i``.
        ``len(result) == n_curves``.

    Raises
    ------
    ValueError
        - Length mismatch (M ≠ N where neither is 1)
        - Unknown dict keys
        - Missing ``'fun'`` key in a dict
        - Invalid top-level type (not str/dict/callable/list/None)
    """
    if fit is None:
        return [[] for _ in range(n_curves)]

    if isinstance(fit, (str, dict)) or callable(fit):
        # Scalar form — broadcast to all curves as a 1-element list
        fit_list = [_normalize_one_fit(fit)]
        return [list(fit_list) for _ in range(n_curves)]

    if isinstance(fit, list):
        normalized = [_normalize_one_fit(f) for f in fit]
        n_fits = len(normalized)
        # Branch order matters: n_fits=n_curves wins over n_fits=1 when both
        # would match (i.e., the n_fits=1, n_curves=1 case goes through here
        # and produces [[fit]] which is correct).
        if n_fits == n_curves:
            # Per-channel pairing: each curve gets its corresponding fit
            return [[normalized[i]] for i in range(n_curves)]
        elif n_curves == 1:
            # Compound: all fits applied to the single curve
            return [list(normalized)]
        elif n_fits == 1:
            # Length-1 list broadcasts (equivalent to scalar form)
            return [list(normalized) for _ in range(n_curves)]
        else:
            raise ValueError(
                f"[dfdraw.fits] fit list length {n_fits} does not match "
                f"n_curves={n_curves}. Vector convention: list length must "
                f"be 1 (broadcast) or equal n_curves. "
                f"Fix: drop a fit, or adjust the expression to match."
            )

    raise ValueError(
        f"[dfdraw.fits] fit= must be str, dict, callable, or list; "
        f"got {type(fit).__name__}. Fix: see Phase 13.42 §3.2."
    )


def _normalize_one_fit(spec) -> Dict:
    """Convert one of {str, callable, dict} to a canonical fit dict."""
    if isinstance(spec, str):
        return {'fun': spec}
    if isinstance(spec, dict):
        # Phase 13.57.DF DR-5: normalize the scipy-vocabulary synonym
        # p0 -> initial BEFORE validation, so downstream code keeps a
        # single canonical key. Conflict (both given, different) raises.
        if 'p0' in spec:
            spec = dict(spec)
            _p0 = spec.pop('p0')
            if 'initial' in spec and list(spec['initial']) != list(_p0):
                raise ValueError(
                    f"[dfdraw.fits] Both 'initial' and its synonym 'p0' "
                    f"were given with different values "
                    f"({spec['initial']!r} vs {_p0!r}). Pass only one "
                    f"(Phase 13.57.DF DR-5)."
                )
            spec['initial'] = _p0
        # Validate dict-key spelling
        unknown = set(spec) - _VALID_DICT_KEYS
        if unknown:
            raise ValueError(
                f"[dfdraw.fits] Unknown key(s) {sorted(unknown)} in fit dict. "
                f"Allowed keys: {sorted(_VALID_DICT_KEYS)}. "
                f"Fix: rename or remove the unknown key."
            )
        if 'fun' not in spec:
            raise ValueError(
                f"[dfdraw.fits] fit dict missing required 'fun' key. "
                f"Fix: add 'fun': '<name>' or 'fun': <callable>."
            )
        return dict(spec)
    if callable(spec):
        return {'fun': spec}
    raise ValueError(
        f"[dfdraw.fits] Each fit must be str, callable, or dict; "
        f"got {type(spec).__name__}."
    )


# =============================================================================
# Single-fit dispatch
# =============================================================================

def dispatch_fit(x_data, y_data, fit_dict, *,
                 yerr=None, plot_kind: str = 'hist') -> Dict[str, Any]:
    """Execute a single fit per the normalized ``fit_dict``.

    Composition (group_by, facet_by, vector) is handled at the caller layer
    by repeated calls to ``dispatch_fit()``.

    Parameters
    ----------
    x_data, y_data : array-like
        Pre-extracted per-curve arrays (see §4.3 of Phase 13.42 proposal).
    fit_dict : dict
        Normalized fit spec (output of ``_normalize_one_fit``).
    yerr : array-like or None
        Per-bin error (used if ``fit_dict['use_errors']`` is True).
    plot_kind : {'hist', 'profile', 'scatter'}
        Used to derive the default value of ``use_errors`` (True for profile,
        False otherwise) per §3.3 of the proposal.

    Returns
    -------
    fit_result : dict
        Per-fit dict per §3.5b of the proposal.
    """
    # ----- Resolve function -----
    fun_spec = fit_dict['fun']
    fit_name: str
    if isinstance(fun_spec, str):
        name = fun_spec.lower()
        if name not in _FIT_REGISTRY:
            raise ValueError(
                f"[dfdraw.fits] Unknown fit name {fun_spec!r}. "
                f"Available: {sorted(_FIT_REGISTRY)}. "
                f"Fix: use a registered name, or register your own with "
                f"dfdraw.register_fit({fun_spec!r}, my_callable)."
            )
        callable_fn = _FIT_REGISTRY[name]
        initial_heuristic = _FIT_INITIAL_HEURISTICS.get(name)
        fit_name = fun_spec
    elif callable(fun_spec):
        callable_fn = fun_spec
        initial_heuristic = None
        fit_name = getattr(fun_spec, '__name__', '<callable>')
    else:
        raise ValueError(
            f"[dfdraw.fits] 'fun' must be str or callable; got "
            f"{type(fun_spec).__name__}."
        )

    # ----- Apply range filter -----
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    yerr_arr = np.asarray(yerr, dtype=float) if yerr is not None else None

    fit_range = fit_dict.get('range')
    if fit_range is not None:
        mask = (x_data >= fit_range[0]) & (x_data <= fit_range[1])
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        yerr_fit = yerr_arr[mask] if yerr_arr is not None else None
    else:
        x_fit, y_fit = x_data, y_data
        yerr_fit = yerr_arr

    # ----- Drop NaN/inf -----
    finite = np.isfinite(x_fit) & np.isfinite(y_fit)
    if yerr_fit is not None:
        finite &= np.isfinite(yerr_fit) & (yerr_fit > 0)
    x_fit = x_fit[finite]
    y_fit = y_fit[finite]
    yerr_fit = yerr_fit[finite] if yerr_fit is not None else None

    # ----- 4-tier initial-param resolution (Phase 13.42 §3.3) -----
    initial = fit_dict.get('initial')
    if initial is not None:
        # Tier 1: user-supplied explicit initial wins absolutely
        p0 = list(initial)
    else:
        user_guess = fit_dict.get('guess')
        if callable(user_guess):
            # Tier 2: user-supplied 'guess' callable
            try:
                p0 = list(user_guess(x_fit, y_fit))
            except Exception as guess_exc:
                warnings.warn(
                    f"[dfdraw.fits] User-supplied 'guess' for fit "
                    f"{fit_name!r} raised {type(guess_exc).__name__}: "
                    f"{guess_exc}. Falling back to registry heuristic.",
                    UserWarning, stacklevel=3)
                p0 = (initial_heuristic(x_fit, y_fit)
                      if initial_heuristic is not None else None)
        elif initial_heuristic is not None:
            # Tier 3: registry heuristic for predefined name
            try:
                p0 = initial_heuristic(x_fit, y_fit)
            except Exception:
                p0 = None
        else:
            # Tier 4: scipy default (all 1's); warn for user callables
            p0 = None
            if not isinstance(fun_spec, str):
                warnings.warn(
                    f"[dfdraw.fits] No 'initial' or 'guess' provided for "
                    f"user callable {fit_name!r}; scipy will use p0=[1]*N. "
                    f"Fix: pass 'initial': [...] or 'guess': callable in "
                    f"the fit dict.",
                    UserWarning, stacklevel=3)

    # ----- Per-bin error weighting -----
    # Phase 13.42.DF FIX1 (B5/R1): use_errors defaults True for both profile AND hist
    # (was: True only for profile). Histogram fit must use Poisson per-bin errors
    # by default — otherwise reported χ² is meaningless (scales with N²). Scatter
    # stays opt-in (its yerr is user-supplied, not auto-Poisson).
    use_errors = fit_dict.get('use_errors', plot_kind in ('profile', 'hist'))
    sigma = yerr_fit if (use_errors and yerr_fit is not None) else None

    # ----- Compose scipy.curve_fit kwargs -----
    cf_kwargs = dict(fit_dict.get('kwargs', {}))
    if 'method' in fit_dict:
        cf_kwargs.setdefault('method', fit_dict['method'])
    if 'bounds' in fit_dict:
        cf_kwargs.setdefault('bounds', fit_dict['bounds'])
    if sigma is not None:
        cf_kwargs.setdefault('absolute_sigma', True)

    # ----- Param-name introspection (for textbox) -----
    try:
        sig = inspect.signature(callable_fn)
        all_params = list(sig.parameters.values())[1:]  # skip x
        param_names = [
            p.name for p in all_params
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                          inspect.Parameter.POSITIONAL_ONLY)
        ]
        if not param_names:
            # Variadic *coeffs (polynomial case): name them c0, c1, ...
            n_params = len(p0) if p0 is not None else 1
            param_names = [f'c{i}' for i in range(n_params)]
    except (ValueError, TypeError):
        param_names = []

    # ----- Execute fit -----
    try:
        if len(x_fit) < max(1, len(param_names)):
            return _failed_fit_dict(
                fit_dict, fit_name, param_names,
                f"insufficient data points "
                f"({len(x_fit)} < {len(param_names)} params)",
            )

        popt, pcov = scipy.optimize.curve_fit(
            callable_fn, x_fit, y_fit, p0=p0, sigma=sigma, **cf_kwargs)

        # Diagnostics
        if not param_names:
            param_names = [f'p{i}' for i in range(len(popt))]
        try:
            with np.errstate(invalid='ignore'):
                perr = np.sqrt(np.diag(pcov))
        except Exception:
            perr = np.full_like(popt, np.nan)

        y_pred = callable_fn(x_fit, *popt)
        residuals = y_fit - y_pred
        if sigma is not None:
            chi2 = float(np.sum((residuals / sigma) ** 2))
        else:
            chi2 = float(np.sum(residuals ** 2))
        ndf = int(max(0, len(x_fit) - len(popt)))
        redchi = float(chi2 / ndf) if ndf > 0 else float('nan')

        threshold = fit_dict.get('redchi_warn_threshold')
        if threshold is not None and np.isfinite(redchi) and redchi > threshold:
            warnings.warn(
                f"[dfdraw.fits] Fit {fit_name!r} redchi {redchi:.2f} exceeds "
                f"threshold {threshold:.2f}. Distribution may not match the "
                f"fitted model; consider quantile-band visualization instead.",
                UserWarning, stacklevel=3)

        def _bound_function(x_eval, _popt=popt, _f=callable_fn):
            return _f(x_eval, *_popt)

        return {
            'fit_name':     fit_name,
            'params':       popt,
            'param_names':  param_names,
            'param_errors': perr,
            'pcov':         pcov,
            'chi2':         chi2,
            'ndf':          ndf,
            'redchi':       redchi,
            'function':     _bound_function,
            'x_range':      (float(x_fit.min()), float(x_fit.max())),
            'n_data':       int(len(x_fit)),
            'fit_status':   'ok',
            'fit_error':    None,
            'fit_spec':     dict(fit_dict),
        }
    except Exception as e:
        if fit_dict.get('raise_on_failure', False):
            raise RuntimeError(
                f"[dfdraw.fits] Fit {fit_name!r} did not converge: {e}. "
                f"Fix: try different initial, narrower range, or a different "
                f"model."
            ) from e
        return _failed_fit_dict(fit_dict, fit_name, param_names, str(e))


def _failed_fit_dict(fit_dict, fit_name, param_names, error_msg) -> Dict[str, Any]:
    return {
        'fit_name':     fit_name,
        'params':       np.array([]),
        'param_names':  param_names,
        'param_errors': np.array([]),
        'pcov':         np.array([[]]),
        'chi2':         float('nan'),
        'ndf':          0,
        'redchi':       float('nan'),
        'function':     None,
        'x_range':      (float('nan'), float('nan')),
        'n_data':       0,
        'fit_status':   'failed',
        'fit_error':    error_msg,
        'fit_spec':     dict(fit_dict),
    }
