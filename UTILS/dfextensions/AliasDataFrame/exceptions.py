"""
Custom exceptions for AliasDataFrame.

Phase 7.4: Define exception hierarchy per architecture review §J1.
"""


class ADFError(Exception):
    """The single root of every ADF-owned error.

    B3.2b STEP 5, deliverable `D_6` §7. Before this, an ADF failure could
    surface as a bare `ValueError` or `NameError` indistinguishable from one
    pandas or numpy raised for an unrelated reason, so calling code could not
    tell "ADF refused, and here is the contract it refused under" from "the
    stack below us broke". `D_6` asks for one root and for the two absence
    kinds to be DISTINGUISHABLE BY TYPE, not only by message text — message
    matching is exactly the coupling that makes error handling rot.

    `AliasDataFrameError` is reparented under this root rather than replaced:
    every existing subclass, every `except AliasDataFrameError` in user code
    and every recorded behaviour keeps working unchanged, and the root is
    added above them instead of beside them.
    """
    pass


class AliasDataFrameError(ADFError):
    """Base exception for AliasDataFrame operations."""
    pass


class StructuralAbsenceError(ADFError):
    """The referenced thing DOES NOT EXIST — `D_6` §7, structural half.

    A subframe, column or function named by an expression is absent from the
    schema entirely. This is a statement about the FRAME, true or false before
    a single row is examined, and no fill can repair it: there is nothing to
    fill from.

    THE SEMANTIC CATEGORY CARRIES NO BUILTIN — v02, GPT29 `F2`. v01 made this
    class a `NameError` and then hung the KeyError leaf beneath it, so
    `S.nosuchcol` became catchable by `except NameError` where it never had
    been. Compatibility is not only about keeping what was caught; widening
    what a handler catches is a change too, and a caller with a broad
    `except NameError` around expression evaluation would silently start
    swallowing subframe column errors it used to let through.

    So each leaf carries exactly the builtin ITS OWN site raised, and the
    shared parent carries none. `b32b_11e` pins both halves as NEGATIVE
    controls, which is the only form of assertion that can catch a widening.
    """
    pass


class ExpressionNameAbsenceError(StructuralAbsenceError, NameError):
    """An expression names something that does not exist — `D_6` §7.

    A column, function or subframe that is simply not there. Also a
    `NameError`, which is what this path raised before and what the condition
    genuinely is. It is deliberately NOT a `KeyError`.
    """
    pass


class RowLevelMissingnessError(ADFError, ValueError):
    """The referenced thing EXISTS but SOME ROWS have no value — `D_6` §7.

    The subframe and column are present and the join is well-formed; specific
    rows simply have no matching key. This is a statement about the DATA, and
    unlike structural absence it IS repairable — a configured fill resolves
    it, which is why ADF refuses rather than inventing one (AD-19).

    The distinction from `StructuralAbsenceError` is the whole point of `D_6`:
    a caller can retry the second with a fill and must never retry the first.

    Also a `ValueError`, which is what this path raised before.
    """
    pass


class ADFProvenanceError(ADFError):
    """Root of the provenance refusals — `D_6` §7, GPT27 `F1`.

    `ADFProvenanceUnsupportedError` predates this hierarchy and inherited only
    `ValueError`, so a caller who adopted `except ADFError` on the strength of
    STEP 5a would have kept missing a known, public ADF refusal. A root that
    does not cover every ADF-owned error is not a root — it is a second
    convention, which is the thing `D_6` exists to remove.

    A branch rather than a direct parent, because provenance is a family: the
    fail-closed row-locality gate is the first member, not the only possible
    one.
    """
    pass


class SubframeColumnAbsenceError(StructuralAbsenceError, KeyError):
    """A subframe EXISTS but does not carry the referenced column — `D_6` §7.

    Structural absence, and the commoner shape of it in practice: `S.nosuchcol`
    rather than `Nope.v`. `b32b_11` drove only the missing-SUBFRAME path, and
    its own docstring warns that a single path silently narrows a criterion —
    so this class exists because that warning was correct. Before STEP 5 this
    path raised a bare `KeyError` and was therefore indistinguishable from a
    pandas lookup failure.

    It is deliberately NOT a `NameError` — see `StructuralAbsenceError`.

    ALSO A `KeyError`, deliberately. The production comment at the raise site
    says in as many words: "Raise KeyError to preserve backward compatibility
    with tests that expect errors on Sub.nonexistent references." That promise
    is kept — `except KeyError` and `pytest.raises(KeyError)` still catch this
    — and ADF ownership is added on top of it. `__str__` is pinned to the
    plain message because `KeyError.__str__` returns `repr(args[0])`, which
    would wrap the message in quotes and break every `match=` that reads it.
    """

    def __str__(self):
        if self.args and isinstance(self.args[0], str):
            return self.args[0]
        return super().__str__()


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
