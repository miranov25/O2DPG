"""
Named Model Registry for Non-Linear Sliding Window Fits

Phase 13.10.GB — provides built-in parametric models, an extensible
registry, and automatic initial parameter estimation.

Each model has:
  - A callable  f(x, *params) → y   (scipy.optimize.curve_fit compatible)
  - A list of parameter names
  - Optional default initial guesses (p0) and bounds
  - Optional p0 estimator: estimate_p0(x, y, weights) → list[float]

Author: Claude13 (Implementer)
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
#  Model dataclass
# ---------------------------------------------------------------------------

class ModelSpec:
    """Specification for a named fit model."""

    __slots__ = ('func', 'param_names', 'default_p0', 'default_bounds',
                 'estimate_p0', 'description')

    def __init__(
        self,
        func: Callable,
        param_names: List[str],
        default_p0: Optional[List[float]] = None,
        default_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
        estimate_p0: Optional[Callable] = None,
        description: str = '',
    ):
        self.func = func
        self.param_names = list(param_names)
        self.default_p0 = list(default_p0) if default_p0 is not None else None
        self.default_bounds = default_bounds
        self.estimate_p0 = estimate_p0
        self.description = description


# ---------------------------------------------------------------------------
#  Built-in model functions (scipy.optimize.curve_fit compatible)
# ---------------------------------------------------------------------------

def _gaussian(x, amplitude, mean, sigma, offset):
    """Gaussian peak + constant offset: A·exp(-(x-μ)²/(2σ²)) + C"""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset


def _polynomial_2(x, coeff_0, coeff_1, coeff_2):
    """Quadratic polynomial: a₀ + a₁x + a₂x²"""
    return coeff_0 + coeff_1 * x + coeff_2 * x ** 2


def _polynomial_3(x, coeff_0, coeff_1, coeff_2, coeff_3):
    """Cubic polynomial: a₀ + a₁x + a₂x² + a₃x³"""
    return coeff_0 + coeff_1 * x + coeff_2 * x ** 2 + coeff_3 * x ** 3


def _exponential(x, amplitude, tau, offset):
    """Exponential decay + offset: A·exp(-x/τ) + C"""
    return amplitude * np.exp(-x / tau) + offset


def _power_law(x, amplitude, exponent, offset):
    """Power law + offset: A·|x|^n + C"""
    return amplitude * np.power(np.abs(x) + 1e-30, exponent) + offset


# ---------------------------------------------------------------------------
#  Built-in p0 estimators
# ---------------------------------------------------------------------------

def _estimate_p0_gaussian(x, y, weights):
    """Estimate Gaussian initial parameters from data.

    Strategy:
      amplitude ≈ max(y) - baseline
      mean      ≈ weighted x near peak
      sigma     ≈ weighted std near peak
      offset    ≈ baseline (median of lowest 20% of y)
    """
    if len(x) < 3:
        return [1.0, 0.0, 1.0, 0.0]

    sorted_y = np.sort(y)
    n_tail = max(1, len(y) // 5)
    offset = float(np.median(sorted_y[:n_tail]))

    y_sub = y - offset
    amplitude = float(np.max(y_sub))
    if amplitude <= 0:
        amplitude = 1.0

    threshold = offset + 0.7 * amplitude
    mask = y >= threshold
    if np.sum(mask) < 2:
        mask = y >= np.median(y)

    w = weights[mask] if weights is not None else np.ones(np.sum(mask))
    w_sum = np.sum(w)
    if w_sum > 0:
        mean = float(np.sum(x[mask] * w) / w_sum)
        sigma = float(np.sqrt(np.sum(w * (x[mask] - mean) ** 2) / w_sum))
    else:
        mean = float(np.mean(x))
        sigma = float(np.std(x))

    sigma = max(sigma, (np.max(x) - np.min(x)) / 100)
    return [amplitude, mean, sigma, offset]


def _estimate_p0_exponential(x, y, weights):
    """Estimate exponential decay initial parameters."""
    if len(x) < 2:
        return [1.0, 1.0, 0.0]
    offset = float(np.min(y))
    amplitude = float(np.max(y) - offset)
    if amplitude <= 0:
        amplitude = 1.0
    x_range = float(np.max(x) - np.min(x))
    tau = max(x_range / 3.0, 1e-10)
    return [amplitude, tau, offset]


def _estimate_p0_power_law(x, y, weights):
    """Estimate power law initial parameters."""
    if len(x) < 2:
        return [1.0, 1.0, 0.0]
    offset = float(np.min(y))
    y_sub = y - offset
    amplitude = float(np.max(y_sub)) if np.max(y_sub) > 0 else 1.0
    return [amplitude, 1.0, offset]


# ---------------------------------------------------------------------------
#  Registry
# ---------------------------------------------------------------------------

_MODEL_REGISTRY: Dict[str, ModelSpec] = {}


def register_fit_model(
    name: str,
    func: Callable,
    param_names: List[str],
    default_p0: Optional[List[float]] = None,
    default_bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    estimate_p0: Optional[Callable] = None,
    description: str = '',
) -> None:
    """Register a named model for non-linear sliding window fits.

    Parameters
    ----------
    name : str
        Model identifier (e.g. ``'crystal_ball'``).
    func : callable
        ``f(x, *params) → y``, scipy.optimize.curve_fit compatible.
    param_names : list of str
        Parameter names matching ``func`` signature (excluding ``x``).
    default_p0 : list of float, optional
        Default initial guesses.
    default_bounds : tuple of (lower, upper), optional
        Parameter bounds for curve_fit.
    estimate_p0 : callable, optional
        ``estimate_p0(x, y, weights) → list[float]`` — data-driven p0.
        Called per bin when no explicit p0 is given.
    description : str, optional
        Human-readable model description.
    """
    if not callable(func):
        raise TypeError(f"func must be callable, got {type(func)}")
    if len(param_names) < 1:
        raise ValueError("param_names must have at least one entry")
    _MODEL_REGISTRY[name] = ModelSpec(
        func=func, param_names=param_names, default_p0=default_p0,
        default_bounds=default_bounds, estimate_p0=estimate_p0,
        description=description,
    )


def get_model(name: str) -> ModelSpec:
    """Look up a named model.  Raises KeyError if not found."""
    if name not in _MODEL_REGISTRY:
        available = sorted(_MODEL_REGISTRY.keys())
        raise KeyError(
            f"Unknown model '{name}'. Available: {available}. "
            f"Use register_fit_model() to add custom models."
        )
    return _MODEL_REGISTRY[name]


def list_models() -> List[str]:
    """Return sorted list of registered model names."""
    return sorted(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
#  Register built-in models
# ---------------------------------------------------------------------------

register_fit_model(
    'gaussian', _gaussian,
    param_names=['amplitude', 'mean', 'sigma', 'offset'],
    default_p0=[1.0, 0.0, 1.0, 0.0],
    default_bounds=([-np.inf, -np.inf, 1e-10, -np.inf],
                    [np.inf, np.inf, np.inf, np.inf]),
    estimate_p0=_estimate_p0_gaussian,
    description='Gaussian peak + constant offset: A·exp(-(x-μ)²/(2σ²)) + C',
)

register_fit_model(
    'polynomial_2', _polynomial_2,
    param_names=['coeff_0', 'coeff_1', 'coeff_2'],
    description='Quadratic polynomial: a₀ + a₁x + a₂x²',
)

register_fit_model(
    'polynomial_3', _polynomial_3,
    param_names=['coeff_0', 'coeff_1', 'coeff_2', 'coeff_3'],
    description='Cubic polynomial: a₀ + a₁x + a₂x² + a₃x³',
)

register_fit_model(
    'exponential', _exponential,
    param_names=['amplitude', 'tau', 'offset'],
    default_p0=[1.0, 1.0, 0.0],
    default_bounds=([-np.inf, 1e-10, -np.inf],
                    [np.inf, np.inf, np.inf]),
    estimate_p0=_estimate_p0_exponential,
    description='Exponential decay + offset: A·exp(-x/τ) + C',
)

register_fit_model(
    'power_law', _power_law,
    param_names=['amplitude', 'exponent', 'offset'],
    default_p0=[1.0, 1.0, 0.0],
    estimate_p0=_estimate_p0_power_law,
    description='Power law + offset: A·|x|^n + C',
)


# ---------------------------------------------------------------------------
#  Gaussian + linear background (most common TPC spectrum model)
# ---------------------------------------------------------------------------

def _gaussian_plus_line(x, amplitude, mean, sigma, offset, slope):
    """Gaussian peak + linear background: A·exp(-(x-μ)²/(2σ²)) + a + bx"""
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset + slope * x


def _estimate_p0_gaussian_plus_line(x, y, weights):
    """Estimate initial params for gaussian + linear background.

    Strategy: estimate linear background from tails, subtract it,
    then estimate Gaussian from residual.
    """
    if len(x) < 5:
        return [1.0, 0.0, 1.0, 0.0, 0.0]

    # Sort by x for tail estimation
    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    # Estimate linear background from lowest/highest 20% of x
    n_tail = max(2, len(x) // 5)
    x_lo, y_lo = x_s[:n_tail], y_s[:n_tail]
    x_hi, y_hi = x_s[-n_tail:], y_s[-n_tail:]

    x_bg = np.concatenate([x_lo, x_hi])
    y_bg = np.concatenate([y_lo, y_hi])

    # Linear fit to background
    if len(x_bg) >= 2 and (np.max(x_bg) - np.min(x_bg)) > 0:
        slope = float((np.mean(x_hi * y_hi) - np.mean(x_lo * y_lo)) /
                       (np.mean(x_hi ** 2) - np.mean(x_lo ** 2) + 1e-30))
        offset = float(np.mean(y_bg) - slope * np.mean(x_bg))
    else:
        slope = 0.0
        offset = float(np.median(y))

    # Subtract background, estimate Gaussian from residual
    y_sub = y - (offset + slope * x)
    amplitude = float(np.max(y_sub))
    if amplitude <= 0:
        amplitude = 1.0

    # Peak position
    threshold = 0.5 * amplitude
    mask = y_sub >= threshold
    if np.sum(mask) < 2:
        mask = y_sub >= np.median(y_sub)

    w = weights[mask] if weights is not None else np.ones(np.sum(mask))
    w_sum = np.sum(w)
    if w_sum > 0:
        mean = float(np.sum(x[mask] * w) / w_sum)
        sigma = float(np.sqrt(np.sum(w * (x[mask] - mean) ** 2) / w_sum))
    else:
        mean = float(np.mean(x))
        sigma = float(np.std(x))

    sigma = max(sigma, (np.max(x) - np.min(x)) / 100)
    return [amplitude, mean, sigma, offset, slope]


register_fit_model(
    'gaussian_plus_line', _gaussian_plus_line,
    param_names=['amplitude', 'mean', 'sigma', 'offset', 'slope'],
    default_p0=[1.0, 0.0, 1.0, 0.0, 0.0],
    default_bounds=([-np.inf, -np.inf, 1e-10, -np.inf, -np.inf],
                    [np.inf, np.inf, np.inf, np.inf, np.inf]),
    estimate_p0=_estimate_p0_gaussian_plus_line,
    description='Gaussian peak + linear background: A·exp(-(x-μ)²/(2σ²)) + a + bx',
)
