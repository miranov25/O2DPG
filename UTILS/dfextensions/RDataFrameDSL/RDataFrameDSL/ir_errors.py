"""
Error handling for RDataFrame DSL.

This module provides structured error types with:
- Error categorization (parse, type, reflection, etc.)
- Source location tracking for helpful error messages
- Suggestions for common mistakes (fuzzy matching)
- Error collection with configurable recovery modes

Error Philosophy:
- Errors should be caught at compile time (before RDF execution)
- Error messages should include context and suggestions
- Multiple errors can be collected and reported together
"""

from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple, Dict, Any
from enum import Enum

__all__ = [
    'IRErrorKind',
    'SourceLocation',
    'IRError',
    'ErrorRecoveryMode',
    'ErrorCollector',
]


class IRErrorKind(Enum):
    """Categories of IR errors."""
    PARSE_ERROR = "parse"           # Python AST parsing failed
    TYPE_ERROR = "type"             # Type mismatch or invalid operation
    RANK_ERROR = "rank"             # Incompatible ranks (scalar vs vector)
    REFLECTION_ERROR = "reflect"    # Method/property not found via ROOT reflection
    UNSUPPORTED_OP = "unsupported"  # Operation not supported in v1
    VALIDATION_ERROR = "validation" # General validation failure
    COMPILE_ERROR = "compile"       # C++ compilation failed
    CYCLE_ERROR = "cycle"           # Circular dependency in aliases
    MISSING_DICT = "dictionary"     # ROOT dictionary not loaded


@dataclass
class SourceLocation:
    """
    Location information for error reporting.
    
    Tracks where in the source expression an error occurred,
    enabling helpful error messages with context.
    
    Attributes:
        expr_name: Name of the alias/expression being processed
        text_span: (start, end) character positions in original text
        line: Line number (if multi-line expression)
        column: Column number
        snippet: Short excerpt of problematic code
    """
    expr_name: str = ""
    text_span: Tuple[int, int] = (0, 0)
    line: Optional[int] = None
    column: Optional[int] = None
    snippet: str = ""
    
    def __str__(self) -> str:
        parts = []
        if self.expr_name:
            parts.append(f"in alias '{self.expr_name}'")
        if self.line is not None:
            loc = f"line {self.line}"
            if self.column is not None:
                loc += f", column {self.column}"
            parts.append(loc)
        if self.snippet:
            parts.append(f"near: \"{self.snippet}\"")
        return ", ".join(parts) if parts else "(unknown location)"


@dataclass
class IRError(Exception):
    """
    Structured error with context for helpful reporting.
    
    IRError captures not just what went wrong, but where and why,
    enabling the DSL to provide actionable error messages.
    
    Attributes:
        kind: Error category (TYPE_ERROR, REFLECTION_ERROR, etc.)
        message: Human-readable error description
        source_location: Where the error occurred
        underlying: Original exception if wrapping another error
        suggestions: List of potential fixes ("Did you mean X?")
        metadata: Additional context (inferred types, etc.)
        
    Example:
        >>> raise IRError(
        ...     IRErrorKind.REFLECTION_ERROR,
        ...     "Method 'getpt' not found in TParticle",
        ...     suggestions=["Did you mean 'GetPt'?"]
        ... )
    """
    kind: IRErrorKind
    message: str
    source_location: Optional[SourceLocation] = None
    underlying: Optional[Exception] = None
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        return self.format_error()
    
    def __repr__(self) -> str:
        return f"IRError({self.kind.name}, {self.message!r})"
    
    def format_error(self, include_suggestions: bool = True) -> str:
        """
        Format error with full context for display.
        
        Args:
            include_suggestions: Whether to include fix suggestions
            
        Returns:
            Formatted multi-line error message
        """
        lines = [f"IRError({self.kind.value}): {self.message}"]
        
        # Source location
        if self.source_location:
            if self.source_location.expr_name:
                lines.append(f"  In alias: '{self.source_location.expr_name}'")
            if self.source_location.snippet:
                lines.append(f"  Near: \"{self.source_location.snippet}\"")
            if self.source_location.line is not None:
                loc = f"  At: line {self.source_location.line}"
                if self.source_location.column is not None:
                    loc += f", column {self.source_location.column}"
                lines.append(loc)
        
        # Type context from metadata
        if 'inferred_type' in self.metadata:
            lines.append(f"  Context:")
            lines.append(f"    • Variable was inferred as: {self.metadata['inferred_type']}")
            if 'type_source' in self.metadata:
                lines.append(f"    • Source: {self.metadata['type_source']}")
        
        if 'expected_type' in self.metadata:
            lines.append(f"    • Expected: {self.metadata['expected_type']}")
        if 'actual_type' in self.metadata:
            lines.append(f"    • Got: {self.metadata['actual_type']}")
        
        # Suggestions
        if include_suggestions and self.suggestions:
            lines.append("  Suggestions:")
            for s in self.suggestions:
                lines.append(f"    • {s}")
        
        # Underlying exception
        if self.underlying:
            lines.append(f"  Caused by: {type(self.underlying).__name__}: {self.underlying}")
        
        return "\n".join(lines)
    
    def with_location(self, location: SourceLocation) -> 'IRError':
        """Return a copy of this error with updated location."""
        return IRError(
            kind=self.kind,
            message=self.message,
            source_location=location,
            underlying=self.underlying,
            suggestions=self.suggestions.copy(),
            metadata=self.metadata.copy()
        )
    
    def with_suggestion(self, suggestion: str) -> 'IRError':
        """Return a copy of this error with an additional suggestion."""
        new_suggestions = self.suggestions.copy()
        new_suggestions.append(suggestion)
        return IRError(
            kind=self.kind,
            message=self.message,
            source_location=self.source_location,
            underlying=self.underlying,
            suggestions=new_suggestions,
            metadata=self.metadata.copy()
        )


class ErrorRecoveryMode(Enum):
    """
    How to handle errors during batch processing.
    
    FAIL_ALL: Stop on first error (strict mode)
    SKIP_CONTINUE: Skip failed alias, continue with others
    FAIL_CHAIN: Fail aliases that depend on failed ones, continue independent
    """
    FAIL_ALL = "fail_all"
    SKIP_CONTINUE = "skip_continue"
    FAIL_CHAIN = "fail_chain"


class ErrorCollector:
    """
    Collects errors with configurable recovery behavior.
    
    Used during batch processing of multiple aliases to:
    - Collect all errors before reporting
    - Track which aliases failed
    - Decide whether to continue processing dependent aliases
    
    Example:
        >>> collector = ErrorCollector(ErrorRecoveryMode.FAIL_CHAIN)
        >>> collector.add(IRError(IRErrorKind.TYPE_ERROR, "Bad type"))
        >>> if collector.should_process("derived", {"base"}):
        ...     # Process 'derived' only if 'base' didn't fail
    """
    
    def __init__(self, mode: ErrorRecoveryMode = ErrorRecoveryMode.FAIL_CHAIN):
        """
        Initialize error collector.
        
        Args:
            mode: Recovery mode determining how to handle errors
        """
        self.mode = mode
        self.errors: List[IRError] = []
        self.failed_aliases: Set[str] = set()
        self._warnings: List[str] = []
    
    def add(self, error: IRError) -> None:
        """
        Add an error to the collection.
        
        Args:
            error: The error to add
        """
        self.errors.append(error)
        
        # Track which alias failed
        if error.source_location and error.source_location.expr_name:
            self.failed_aliases.add(error.source_location.expr_name)
    
    def add_warning(self, message: str) -> None:
        """Add a warning (non-fatal)."""
        self._warnings.append(message)
    
    def should_process(self, alias: str, dependencies: Set[str]) -> bool:
        """
        Determine if an alias should be processed based on error state.
        
        Args:
            alias: Name of alias to potentially process
            dependencies: Set of alias names this alias depends on
            
        Returns:
            True if the alias should be processed
        """
        if self.mode == ErrorRecoveryMode.FAIL_ALL:
            # Stop everything if any error occurred
            return len(self.errors) == 0
        
        elif self.mode == ErrorRecoveryMode.SKIP_CONTINUE:
            # Only skip if this specific alias failed
            return alias not in self.failed_aliases
        
        elif self.mode == ErrorRecoveryMode.FAIL_CHAIN:
            # Skip if any dependency failed
            return not bool(dependencies & self.failed_aliases)
        
        return True
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if any warnings were collected."""
        return len(self._warnings) > 0
    
    def error_count(self) -> int:
        """Return number of errors."""
        return len(self.errors)
    
    def clear(self) -> None:
        """Clear all collected errors and warnings."""
        self.errors.clear()
        self.failed_aliases.clear()
        self._warnings.clear()
    
    def format_report(self, include_warnings: bool = True) -> str:
        """
        Format all collected errors into a report.
        
        Args:
            include_warnings: Whether to include warnings
            
        Returns:
            Formatted multi-line report string
        """
        if not self.errors and not (include_warnings and self._warnings):
            return "No errors."
        
        lines = []
        
        # Errors
        if self.errors:
            lines.append(f"Found {len(self.errors)} error(s):")
            for i, error in enumerate(self.errors, 1):
                lines.append(f"\n{i}. {error.format_error()}")
        
        # Warnings
        if include_warnings and self._warnings:
            if lines:
                lines.append("")
            lines.append(f"Warnings ({len(self._warnings)}):")
            for warning in self._warnings:
                lines.append(f"  • {warning}")
        
        # Summary of failed aliases
        if self.failed_aliases:
            lines.append(f"\nFailed aliases: {', '.join(sorted(self.failed_aliases))}")
        
        return "\n".join(lines)
    
    def raise_if_errors(self) -> None:
        """Raise the first error if any errors were collected."""
        if self.errors:
            raise self.errors[0]
    
    def get_errors_by_kind(self, kind: IRErrorKind) -> List[IRError]:
        """Get all errors of a specific kind."""
        return [e for e in self.errors if e.kind == kind]


# =============================================================================
# Helper Functions for Creating Common Errors
# =============================================================================

def type_mismatch_error(
    expected: str,
    actual: str,
    context: str = "",
    location: Optional[SourceLocation] = None
) -> IRError:
    """Create a type mismatch error with helpful context."""
    msg = f"Type mismatch: expected {expected}, got {actual}"
    if context:
        msg += f" ({context})"
    
    return IRError(
        kind=IRErrorKind.TYPE_ERROR,
        message=msg,
        source_location=location,
        metadata={'expected_type': expected, 'actual_type': actual}
    )


def unknown_variable_error(
    name: str,
    location: Optional[SourceLocation] = None,
    similar_names: List[str] = None
) -> IRError:
    """Create an unknown variable error with suggestions."""
    suggestions = []
    if similar_names:
        for similar in similar_names[:3]:
            suggestions.append(f"Did you mean '{similar}'?")
    
    return IRError(
        kind=IRErrorKind.TYPE_ERROR,
        message=f"Unknown variable '{name}' - not found in tree or aliases",
        source_location=location,
        suggestions=suggestions
    )


def method_not_found_error(
    class_name: str,
    method_name: str,
    location: Optional[SourceLocation] = None,
    similar_methods: List[str] = None
) -> IRError:
    """Create a method not found error with suggestions."""
    suggestions = []
    if similar_methods:
        for similar in similar_methods[:3]:
            suggestions.append(f"Did you mean '{similar}'?")
    suggestions.append("Check if the dictionary is loaded")
    
    return IRError(
        kind=IRErrorKind.REFLECTION_ERROR,
        message=f"Method '{method_name}' not found in class '{class_name}'",
        source_location=location,
        suggestions=suggestions
    )


def property_not_found_error(
    class_name: str,
    property_name: str,
    location: Optional[SourceLocation] = None
) -> IRError:
    """Create a property not found error."""
    return IRError(
        kind=IRErrorKind.REFLECTION_ERROR,
        message=f"Property '{property_name}' not found in class '{class_name}'",
        source_location=location,
        suggestions=[
            "Check if the property is public",
            "Check if the dictionary is loaded"
        ]
    )


def missing_dictionary_error(
    class_name: str,
    location: Optional[SourceLocation] = None
) -> IRError:
    """Create a missing dictionary error."""
    return IRError(
        kind=IRErrorKind.MISSING_DICT,
        message=f"Class '{class_name}' not found. Dictionary may not be loaded.",
        source_location=location,
        suggestions=[
            "Load dictionary: ROOT.gInterpreter.ProcessLine('.L dict.C+')",
            "Or provide schema override with cpp_type"
        ]
    )


def rank_mismatch_error(
    operation: str,
    left_rank: int,
    right_rank: int,
    location: Optional[SourceLocation] = None
) -> IRError:
    """Create a rank mismatch error."""
    return IRError(
        kind=IRErrorKind.RANK_ERROR,
        message=f"Rank mismatch in {operation}: rank {left_rank} vs rank {right_rank}",
        source_location=location,
        metadata={'left_rank': left_rank, 'right_rank': right_rank}
    )


def unsupported_operation_error(
    operation: str,
    reason: str = "",
    location: Optional[SourceLocation] = None
) -> IRError:
    """Create an unsupported operation error."""
    msg = f"Unsupported operation: {operation}"
    if reason:
        msg += f" ({reason})"
    
    return IRError(
        kind=IRErrorKind.UNSUPPORTED_OP,
        message=msg,
        source_location=location
    )


def compile_error(
    function_name: str,
    cpp_error: str,
    location: Optional[SourceLocation] = None
) -> IRError:
    """Create a C++ compilation error."""
    return IRError(
        kind=IRErrorKind.COMPILE_ERROR,
        message=f"Failed to compile helper function '{function_name}'",
        source_location=location,
        suggestions=[
            "Check the generated C++ code for syntax errors",
            "Ensure all required headers are included"
        ],
        metadata={'cpp_error': cpp_error}
    )
