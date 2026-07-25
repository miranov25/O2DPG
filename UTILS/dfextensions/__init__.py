"""
dfextensions - DataFrame extensions and utilities.

Main packages:
- AliasDataFrame: Lazy-evaluated DataFrame with compression support
- groupby_regression: Grouped regression utilities
- quantile_fit_nd: N-dimensional quantile fitting
- dataframe_utils: Plotting and statistics utilities
- formula_utils: Formula-based modeling and code export
- diagnostics: Host and job diagnostics for calibration workloads

Public API and __version__ are preserved from the pre-PHASE_13_74 umbrella.
The heavy analysis packages are exposed LAZILY (PEP 562): `from dfextensions
import AliasDataFrame` still works on demand, but merely importing a light
subpackage - e.g. `import dfextensions.diagnostics` for collection-only use -
does NOT eagerly pull in the analysis stack. This keeps the umbrella's public
surface intact (restoring what PHASE_13_74 Increment 1 had inadvertently
dropped) while allowing the diagnostics collector to import without ADF/dfdraw
present [PHASE_13_74_ADF D0 / T-P5].
"""
__version__ = '1.1.0'

__all__ = [
    "AliasDataFrame",
    "CompressionState",
    "FormulaLinearModel",
    "GroupByRegressor",     # from groupby_regression
]

# Lazy public API [PEP 562]: same names as the original eager umbrella, resolved
# on first access so a light subpackage import does not drag in the analysis
# stack. Mapping: exported name -> (submodule, attribute-or-None-for-star).
_LAZY = {
    "AliasDataFrame":    (".AliasDataFrame", "AliasDataFrame"),
    "CompressionState":  (".AliasDataFrame", "CompressionState"),
    "FormulaLinearModel": (".formula_utils", "FormulaLinearModel"),
    "GroupByRegressor":  (".groupby_regression", "GroupByRegressor"),
}


def __getattr__(name):          # PEP 562 module-level lazy attribute
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    mod = importlib.import_module(target[0], __name__)
    val = getattr(mod, target[1])
    globals()[name] = val       # cache: subsequent access skips __getattr__
    return val


def __dir__():
    return sorted(list(globals().keys()) + __all__)
