"""dfextensions.diagnostics - host and job diagnostics for calibration workloads.

Answers three questions from time-stamped measurement series:
  1. is this machine healthy?
  2. did the machine's load affect my job?
  3. did my job affect the machine?  (recorded as required; not yet implemented)

Canonical imports
-----------------
    from dfextensions.diagnostics import RunMetrics
    from dfextensions.diagnostics import report_diagnostics

Both work from any working directory once the parent of ``dfextensions`` is
importable.  Direct script invocation of the modules themselves remains
supported and unchanged:

    python report_diagnostics.py <bundle> -o <out>

Design note (Increment 1): this package performs NO manipulation of
``sys.path``.  Submodules resolve their siblings by package-relative import
when imported as part of this package, and by plain import when run directly
as scripts - in the latter case Python itself places the script's directory on
the search path, so no manual insertion is required.

The report layer is exposed lazily: importing this package does NOT import
``report_diagnostics``, so the collection core stays importable on hosts where
the optional AliasDataFrame/dfdraw analysis stack is absent.

[PHASE_13_74_ADF Increment 1, deliverables D0 / S-20]
"""
from __future__ import annotations

import sys

# --- supported interpreter floor -------------------------------------------
# Architect confirmation A-3: the supported Python floor is 3.10.  Enforced at
# import time with a named, actionable error rather than an obscure syntax or
# attribute failure deep inside a submodule.
MIN_PYTHON = (3, 10)
if sys.version_info < MIN_PYTHON:
    raise RuntimeError(
        "dfextensions.diagnostics requires Python >= "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}; this interpreter is "
        f"{sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]} "
        f"at {sys.executable}"
    )

__all__ = ["RunMetrics", "report_diagnostics", "MIN_PYTHON"]

# --- eager: collection core (no optional dependencies) ----------------------
from .run_metrics import RunMetrics  # noqa: E402,F401


def __getattr__(name):
    """Lazy attribute access [PEP 562].

    ``report_diagnostics`` pulls in the analysis/rendering stack, which is
    optional.  Importing it here would make the whole package unimportable on
    a collection-only host, so it is resolved on first use instead.
    """
    if name == "report_diagnostics":
        # importlib, NOT `from . import ...`: the latter re-enters this very
        # __getattr__ and recurses without bound.  import_module also binds the
        # submodule onto this package, so later attribute access and
        # `from dfextensions.diagnostics import report_diagnostics` resolve
        # directly and never reach __getattr__ again.
        import importlib
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
