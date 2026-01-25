"""
Verbosity constants for DSLCompiler.describe_structure()

Phase 13.6.G: Interactive inspection with bitmask-based verbosity control.

Design follows AliasDataFrame pattern for consistency across projects.

Usage:
    from RDataFrameDSL.verbosity import VERBOSE_DEFAULT, VERBOSE_FULL
    
    dsl = DSLCompiler.from_rdf(rdf)
    dsl.describe_structure()                    # Default verbosity
    dsl.describe_structure(VERBOSE_FULL)        # Everything
    dsl.describe_structure(VERBOSITY_ALIASES)   # Just aliases
    
    # Programmatic access
    info = dsl.describe_structure(return_dict=True)

Author: Claude (Phase 13.6.G)
Date: 2026-01-25
"""

# =============================================================================
# Verbosity Bitmask Constants
# =============================================================================

VERBOSITY_BASIC        = 0x01  # Row count, column count, memory estimate
VERBOSITY_SCHEMA       = 0x02  # Column names and types
VERBOSITY_ALIASES      = 0x04  # Alias pool (names only)
VERBOSITY_ALIASES_FULL = 0x08  # Alias expressions
VERBOSITY_DEFINES      = 0x10  # Defined columns (compiled)
VERBOSITY_COMPILED     = 0x20  # Compilation status details
VERBOSITY_SAFE_MODE    = 0x40  # Safe mode configuration
VERBOSITY_CACHE        = 0x80  # Cache statistics

# =============================================================================
# Presets
# =============================================================================

VERBOSE_MINIMAL = VERBOSITY_BASIC
"""Minimal output: just row/column counts."""

VERBOSE_DEFAULT = (
    VERBOSITY_BASIC | 
    VERBOSITY_SCHEMA | 
    VERBOSITY_ALIASES | 
    VERBOSITY_DEFINES
)
"""Default output: basic info, schema, aliases, and defined columns."""

VERBOSE_FULL = 0xFF
"""Full output: all available information."""

# Alias for convenience
VERBOSE_ALL = VERBOSE_FULL

# =============================================================================
# Flag Names (for introspection)
# =============================================================================

FLAG_NAMES = {
    VERBOSITY_BASIC: 'VERBOSITY_BASIC',
    VERBOSITY_SCHEMA: 'VERBOSITY_SCHEMA',
    VERBOSITY_ALIASES: 'VERBOSITY_ALIASES',
    VERBOSITY_ALIASES_FULL: 'VERBOSITY_ALIASES_FULL',
    VERBOSITY_DEFINES: 'VERBOSITY_DEFINES',
    VERBOSITY_COMPILED: 'VERBOSITY_COMPILED',
    VERBOSITY_SAFE_MODE: 'VERBOSITY_SAFE_MODE',
    VERBOSITY_CACHE: 'VERBOSITY_CACHE',
}


def describe_flags(verbosity: int) -> str:
    """
    Return human-readable description of verbosity flags.
    
    Args:
        verbosity: Bitmask of VERBOSITY_* flags
        
    Returns:
        String listing active flags
        
    Example:
        >>> describe_flags(VERBOSE_DEFAULT)
        'VERBOSITY_BASIC | VERBOSITY_SCHEMA | VERBOSITY_ALIASES | VERBOSITY_DEFINES'
    """
    active = []
    for flag, name in FLAG_NAMES.items():
        if verbosity & flag:
            active.append(name)
    return ' | '.join(active) if active else 'NONE'


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Flags
    'VERBOSITY_BASIC',
    'VERBOSITY_SCHEMA',
    'VERBOSITY_ALIASES',
    'VERBOSITY_ALIASES_FULL',
    'VERBOSITY_DEFINES',
    'VERBOSITY_COMPILED',
    'VERBOSITY_SAFE_MODE',
    'VERBOSITY_CACHE',
    # Presets
    'VERBOSE_MINIMAL',
    'VERBOSE_DEFAULT',
    'VERBOSE_FULL',
    'VERBOSE_ALL',
    # Helpers
    'FLAG_NAMES',
    'describe_flags',
]
