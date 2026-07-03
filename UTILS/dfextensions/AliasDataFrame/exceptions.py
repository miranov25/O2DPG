"""
Custom exceptions for AliasDataFrame.

Phase 7.4: Define exception hierarchy per architecture review §J1.
"""


class AliasDataFrameError(Exception):
    """Base exception for AliasDataFrame operations."""
    pass


class BranchNotFoundError(AliasDataFrameError, ValueError):
    """Branch not found in TTree.
    
    Attributes
    ----------
    missing : set
        Branches that were requested but not found
    available : set
        Branches that are available
    """
    def __init__(self, missing: set, available: set = None, message: str = None):
        self.missing = missing
        self.available = available
        if message is None:
            message = f"Branches not found: {sorted(missing)}"
            if available:
                avail_list = sorted(available)[:10]
                suffix = "..." if len(available) > 10 else ""
                message += f". Available: {avail_list}{suffix}"
        super().__init__(message)


class ChainMetadataCompatibilityError(AliasDataFrameError):
    """PHASE_13_67_ADF: chain files carry incompatible ADF metadata (aliases, schema/
    dtypes, subframe definitions, or compression entries). Names the offending file
    index, path, and item. Raised by strict metadata validation over a lazy chain."""
    pass


class ChainValidationError(AliasDataFrameError):
    """Chain validation failed due to branch mismatch.
    
    Attributes
    ----------
    file_idx : int
        Index of file that failed validation
    file_path : str
        Path to file that failed
    expected : set
        Expected branches (from reference file)
    actual : set
        Actual branches in failing file
    """
    def __init__(self, file_idx: int, file_path: str, 
                 expected: set, actual: set, message: str = None):
        self.file_idx = file_idx
        self.file_path = file_path
        self.expected = expected
        self.actual = actual
        if message is None:
            missing = expected - actual
            extra = actual - expected
            message = f"Branch mismatch in file {file_idx} ({file_path})"
            if missing:
                message += f". Missing: {sorted(missing)}"
            if extra:
                message += f". Extra: {sorted(extra)}"
        super().__init__(message)


class CircularAliasError(AliasDataFrameError):
    """Circular dependency detected in alias definitions.
    
    Note: Defined for future use. Currently alias cycle detection
    raises ValueError. Will be wired in a future phase.
    
    Attributes
    ----------
    cycle : list
        List of alias names forming the cycle
    """
    def __init__(self, cycle: list, message: str = None):
        self.cycle = cycle
        if message is None:
            message = f"Circular alias dependency: {' → '.join(cycle)}"
        super().__init__(message)
