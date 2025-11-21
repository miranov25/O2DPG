"""
AliasDataFrame - Lazy-evaluated DataFrame with compression support.

Main exports:
- AliasDataFrame: Main class
- CompressionState: State class for compression tracking
"""

"""
AliasDataFrame - Lazy-evaluated DataFrame with compression support.

Main exports:
- AliasDataFrame: Main class
- CompressionState: State class for compression tracking
- Verbosity constants for describe_structure()
"""

from .AliasDataFrame import (
    AliasDataFrame,
    CompressionState,
    # Verbosity bitmask constants (Phase 2)
    VERBOSITY_BASIC,
    VERBOSITY_DTYPES,
    VERBOSITY_ALIASES,
    VERBOSITY_ALIASES_FULL,
    VERBOSITY_COMPRESSION,
    VERBOSITY_COMP_FULL,
    VERBOSITY_SUBFRAMES,
    VERBOSITY_METADATA,
    # Verbosity presets
    VERBOSE_MINIMAL,
    VERBOSE_DEFAULT,
    VERBOSE_FULL,
)

__all__ = [
    'AliasDataFrame',
    'CompressionState',
    # Verbosity constants
    'VERBOSITY_BASIC',
    'VERBOSITY_DTYPES',
    'VERBOSITY_ALIASES',
    'VERBOSITY_ALIASES_FULL',
    'VERBOSITY_COMPRESSION',
    'VERBOSITY_COMP_FULL',
    'VERBOSITY_SUBFRAMES',
    'VERBOSITY_METADATA',
    'VERBOSE_MINIMAL',
    'VERBOSE_DEFAULT',
    'VERBOSE_FULL',
]

