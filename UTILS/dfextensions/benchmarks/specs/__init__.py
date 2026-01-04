"""
Benchmark Framework v1.0 — Visualization Specs Loader

Phase 12.14c.GB D2: Load benchmark visualization specifications from YAML.

Public API:
    load_benchmark_specs(specs_path) -> list[dict]

Private:
    _validate_specs(specs) -> None

Key constraints (from review):
    P1-1: PyYAML try/except with install message
    P1-3: Spec schema validation (fail-closed)
    P1-10: Schema version in YAML
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# SCHEMA VALIDATION
# =============================================================================

# Required keys by plot kind
REQUIRED_BY_KIND = {
    'scatter': ['name', 'x', 'y'],
    'line': ['name', 'x', 'y'],
    'hist': ['name', 'x'],
    'bar': ['name', 'y'],
}
ALLOWED_KINDS = set(REQUIRED_BY_KIND.keys())


def _validate_specs(specs: list[dict]) -> None:
    """
    Validate spec schema. Fail-closed on errors.
    
    Parameters
    ----------
    specs : list[dict]
        List of spec dictionaries to validate
    
    Raises
    ------
    ValueError
        If any spec is missing required keys or has invalid kind
    """
    for i, spec in enumerate(specs):
        # Name is always required
        if 'name' not in spec:
            raise ValueError(f"Spec #{i} missing required key 'name': {spec}")
        
        name = spec['name']
        kind = spec.get('kind', 'scatter')
        
        if kind not in ALLOWED_KINDS:
            raise ValueError(
                f"Spec '{name}': invalid kind '{kind}'. "
                f"Allowed: {ALLOWED_KINDS}"
            )
        
        required = REQUIRED_BY_KIND[kind]
        for key in required:
            if key not in spec:
                raise ValueError(
                    f"Spec '{name}' (kind={kind}) missing required key '{key}'"
                )


# =============================================================================
# PUBLIC API
# =============================================================================

def load_benchmark_specs(specs_path: Optional[Path] = None) -> list[dict]:
    """
    Load visualization specifications from YAML.
    
    NOTE: specs_path parameter is INTERNAL. No public CLI flag in Phase 12.14c.
    
    Parameters
    ----------
    specs_path : Path, optional
        Custom specs file. Default: benchmarks/specs/benchmark_specs.yaml
    
    Returns
    -------
    list[dict]
        List of spec dictionaries with keys:
        name, title, x, y, kind, groupby, filter, enabled, bins
    
    Raises
    ------
    ImportError
        If PyYAML not installed
    ValueError
        If spec missing required keys
    FileNotFoundError
        If specs file not found
    
    Examples
    --------
    >>> specs = load_benchmark_specs()
    >>> len(specs)
    7
    >>> specs[0]['name']
    'time_trend'
    """
    # P1-1: PyYAML handling
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML required for benchmark specs. "
            "Install with: pip install pyyaml"
        )
    
    if specs_path is None:
        specs_path = Path(__file__).parent / "benchmark_specs.yaml"
    
    specs_path = Path(specs_path)
    
    if not specs_path.exists():
        raise FileNotFoundError(f"Specs file not found: {specs_path}")
    
    with open(specs_path) as f:
        data = yaml.safe_load(f)
    
    if data is None:
        logger.warning(f"Empty specs file: {specs_path}")
        return []
    
    specs = data.get("specs", [])
    
    # P1-3: Schema validation (fail-closed)
    _validate_specs(specs)
    
    # Log schema version if present
    meta = data.get("__meta__", {})
    schema_version = meta.get("schema_version")
    if schema_version:
        logger.debug(f"Loaded specs schema version: {schema_version}")
    
    return specs


def get_enabled_specs(specs: Optional[list[dict]] = None) -> list[dict]:
    """
    Get only enabled specs.
    
    Parameters
    ----------
    specs : list[dict], optional
        Specs to filter. Default: load from default path
    
    Returns
    -------
    list[dict]
        List of specs where enabled=True (or not specified)
    """
    if specs is None:
        specs = load_benchmark_specs()
    
    return [s for s in specs if s.get('enabled', True)]
