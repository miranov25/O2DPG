"""
IR node classes for RDataFrame DSL.

This module defines the Intermediate Representation (IR) node classes
that represent parsed expressions. The IR is:
- Independent of Python AST (can be constructed manually)
- Type-annotated (each node knows its result type and rank)
- Walkable (supports tree traversal for analysis and code generation)

Node Hierarchy:
- IRNode (base class)
  - ConstantNode (literal values)
  - VariableNode (column/alias references)
  - UnaryOpNode (negation, not, etc.)
  - BinaryOpNode (arithmetic, comparison, logical)
  - TernaryOpNode (conditional expression)
  - CallNode (global function calls)
  - MethodCallNode (method calls on objects)
  - PropertyAccessNode (property/field access)
  - SubscriptNode (indexing and slicing)
  - SliceNode (slice parameters)
  - CollectionIndexNode (cross-collection indexing)

Rank System:
- rank=0: Scalar value
- rank=1: 1D vector (RVec<T>)
- rank=2: 2D nested vector (RVec<RVec<T>>)

Jaggedness:
- is_jagged=True: Inner vectors may have different lengths
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Iterator
from enum import Enum

from .ir_types import IRType, IRTypeKind
from .ir_errors import SourceLocation

__all__ = [
    # Enums
    'UnaryOp',
    'BinaryOp',
    'SliceKind',
    # Base
    'IRNode',
    # Leaf nodes
    'ConstantNode',
    'VariableNode',
    # Operator nodes
    'UnaryOpNode',
    'BinaryOpNode',
    'TernaryOpNode',
    # Function/method nodes
    'CallNode',
    'MethodCallNode',
    'PropertyAccessNode',
    # Broadcasting nodes (Phase 8)
    'MethodBroadcastNode',
    'PropertyBroadcastNode',
    # Indexing nodes
    'SliceNode',
    'SubscriptNode',
    'CollectionIndexNode',
    'RVecSliceNode',
    # Helpers
    'BroadcastInfo',
]


# =============================================================================
# Operator Enums
# =============================================================================

class UnaryOp(Enum):
    """Unary operators."""
    NEG = "-"       # Arithmetic negation
    POS = "+"       # Arithmetic positive (usually no-op)
    NOT = "not"     # Logical not
    BITNOT = "~"    # Bitwise not
    
    def to_cpp(self) -> str:
        """Return C++ operator string."""
        if self == UnaryOp.NEG:
            return "-"
        elif self == UnaryOp.POS:
            return "+"
        elif self == UnaryOp.NOT:
            return "!"
        elif self == UnaryOp.BITNOT:
            return "~"
        return str(self.value)


class BinaryOp(Enum):
    """Binary operators."""
    # Arithmetic
    ADD = "+"
    SUB = "-"
    MUL = "*"
    DIV = "/"
    FLOORDIV = "//"
    MOD = "%"
    POW = "**"
    
    # Comparison
    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="
    
    # Logical
    AND = "and"
    OR = "or"
    
    # Bitwise
    BITAND = "&"
    BITOR = "|"
    BITXOR = "^"
    
    def is_arithmetic(self) -> bool:
        """Check if operator is arithmetic."""
        return self in (
            BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL,
            BinaryOp.DIV, BinaryOp.FLOORDIV, BinaryOp.MOD, BinaryOp.POW
        )
    
    def is_comparison(self) -> bool:
        """Check if operator is comparison."""
        return self in (
            BinaryOp.LT, BinaryOp.LE, BinaryOp.GT,
            BinaryOp.GE, BinaryOp.EQ, BinaryOp.NE
        )
    
    def is_logical(self) -> bool:
        """Check if operator is logical."""
        return self in (BinaryOp.AND, BinaryOp.OR)
    
    def is_bitwise(self) -> bool:
        """Check if operator is bitwise."""
        return self in (BinaryOp.BITAND, BinaryOp.BITOR, BinaryOp.BITXOR)
    
    def to_cpp(self) -> str:
        """Return C++ operator string."""
        cpp_ops = {
            BinaryOp.ADD: "+",
            BinaryOp.SUB: "-",
            BinaryOp.MUL: "*",
            BinaryOp.DIV: "/",
            BinaryOp.FLOORDIV: "/",  # Will need special handling
            BinaryOp.MOD: "%",
            BinaryOp.POW: "pow",     # Function call
            BinaryOp.LT: "<",
            BinaryOp.LE: "<=",
            BinaryOp.GT: ">",
            BinaryOp.GE: ">=",
            BinaryOp.EQ: "==",
            BinaryOp.NE: "!=",
            BinaryOp.AND: "&&",
            BinaryOp.OR: "||",
            BinaryOp.BITAND: "&",
            BinaryOp.BITOR: "|",
            BinaryOp.BITXOR: "^",
        }
        return cpp_ops.get(self, str(self.value))


class SliceKind(Enum):
    """
    Classification of slice operations for code generation.
    
    Each kind maps to a specific C++ code generation pattern.
    """
    FIRST_N = "first_n"       # [:3]      → Take(v, 3)
    LAST_N = "last_n"         # [-3:]     → Take(v, -3)
    FROM_INDEX = "from_index" # [2:]      → Take(v, Range(2, size))
    RANGE = "range"           # [1:3]     → Take(v, Range(1, 3))
    RANGE_NEG = "range_neg"   # [1:-1]    → Range with negative index translation
    STEP = "step"             # [::2]     → loop-based indices
    REVERSE = "reverse"       # [::-1]    → manual reverse loop
    BOOLEAN = "boolean"       # [mask]    → native v[mask]


# =============================================================================
# Helper Classes
# =============================================================================

@dataclass
class BroadcastInfo:
    """
    Information about broadcasting in binary operations.
    
    When combining arrays of different ranks, broadcasting rules apply:
    - Scalar (rank 0) broadcasts to any rank
    - Rank-k op Rank-k requires matching dimensions
    
    Attributes:
        left_broadcast: Whether left operand needs broadcasting
        right_broadcast: Whether right operand needs broadcasting
        result_rank: Resulting rank after broadcast
    """
    left_broadcast: bool = False
    right_broadcast: bool = False
    result_rank: int = 0


# =============================================================================
# Base Node
# =============================================================================

@dataclass
class IRNode:
    """
    Base class for all IR nodes.
    
    Every IR node carries:
    - dtype: The result type of this expression
    - rank: Dimensionality (0=scalar, 1=vector, 2=matrix)
    - is_jagged: Whether inner dimensions vary in size
    - source_location: Where in source this came from (for errors)
    - metadata: Additional analysis information
    
    Subclasses should override children() to enable tree traversal.
    """
    dtype: IRType = field(default_factory=lambda: IRType(IRTypeKind.Unknown))
    rank: int = 0
    is_jagged: bool = False
    source_location: Optional[SourceLocation] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def children(self) -> List['IRNode']:
        """
        Return child nodes for tree traversal.
        
        Override in subclasses that have children.
        """
        return []
    
    def walk(self) -> Iterator['IRNode']:
        """
        Iterate over all nodes in subtree (pre-order traversal).
        
        Yields:
            Each node in the subtree, starting with self
        """
        yield self
        for child in self.children():
            if child is not None:
                yield from child.walk()
    
    def walk_postorder(self) -> Iterator['IRNode']:
        """
        Iterate over all nodes in subtree (post-order traversal).
        
        Useful for bottom-up processing (e.g., type inference).
        """
        for child in self.children():
            if child is not None:
                yield from child.walk_postorder()
        yield self
    
    def depth(self) -> int:
        """Return depth of the expression tree."""
        child_depths = [c.depth() for c in self.children() if c is not None]
        return 1 + max(child_depths) if child_depths else 1
    
    def node_count(self) -> int:
        """Return total number of nodes in subtree."""
        return sum(1 for _ in self.walk())
    
    def collect_variables(self) -> List['VariableNode']:
        """Collect all variable references in subtree."""
        return [n for n in self.walk() if isinstance(n, VariableNode)]
    
    def collect_calls(self) -> List['CallNode']:
        """Collect all function calls in subtree."""
        return [n for n in self.walk() if isinstance(n, CallNode)]
    
    def has_method_calls(self) -> bool:
        """Check if subtree contains any method calls."""
        return any(isinstance(n, MethodCallNode) for n in self.walk())


# =============================================================================
# Leaf Nodes
# =============================================================================

@dataclass
class ConstantNode(IRNode):
    """
    Literal constant value.
    
    Represents Python literals: int, float, bool.
    Type is inferred from the Python value type.
    
    Attributes:
        value: The literal value
        
    Examples:
        >>> ConstantNode(value=42)      # Int64
        >>> ConstantNode(value=3.14)    # Float64
        >>> ConstantNode(value=True)    # Bool
    """
    value: Union[int, float, bool] = 0
    
    def __post_init__(self):
        # Infer type from value
        if isinstance(self.value, bool):
            self.dtype = IRType(IRTypeKind.Bool)
        elif isinstance(self.value, int):
            self.dtype = IRType(IRTypeKind.Int64)
        elif isinstance(self.value, float):
            self.dtype = IRType(IRTypeKind.Float64)
        else:
            self.dtype = IRType(IRTypeKind.Unknown)
        
        # Constants are always scalar
        self.rank = 0
        self.is_jagged = False
    
    def __repr__(self) -> str:
        return f"ConstantNode({self.value!r}, dtype={self.dtype})"
    
    def to_cpp(self) -> str:
        """Return C++ literal representation."""
        if isinstance(self.value, bool):
            return "true" if self.value else "false"
        elif isinstance(self.value, float):
            # Ensure float literal in C++
            s = repr(self.value)
            if 'e' not in s.lower() and '.' not in s:
                s += '.0'
            return s
        else:
            return str(self.value)


@dataclass
class VariableNode(IRNode):
    """
    Reference to a column, alias, or subframe column.
    
    Attributes:
        name: Variable/column name
        namespace: Subframe name (e.g., "calib" in "calib.offset")
        is_alias: Whether this references a computed alias
        cpp_type: Known C++ type (from reflection)
        
    Examples:
        >>> VariableNode(name="px")                    # Main tree column
        >>> VariableNode(name="offset", namespace="calib")  # Subframe column
        >>> VariableNode(name="pt", is_alias=True)    # Computed alias
    """
    name: str = ""
    namespace: Optional[str] = None
    is_alias: bool = False
    cpp_type: Optional[str] = None
    
    def __repr__(self) -> str:
        parts = [f"VariableNode({self.name!r}"]
        if self.namespace:
            parts.append(f", namespace={self.namespace!r}")
        if self.is_alias:
            parts.append(", is_alias=True")
        parts.append(f", dtype={self.dtype}, rank={self.rank})")
        return "".join(parts)
    
    def full_name(self) -> str:
        """Return fully qualified name including namespace."""
        if self.namespace:
            return f"{self.namespace}.{self.name}"
        return self.name
    
    def to_cpp(self) -> str:
        """Return C++ variable reference."""
        return self.full_name()


# =============================================================================
# Operator Nodes
# =============================================================================

@dataclass
class UnaryOpNode(IRNode):
    """
    Unary operation (negation, logical not, bitwise not).
    
    Attributes:
        op: The unary operator
        operand: The operand expression
        
    Example:
        >>> UnaryOpNode(op=UnaryOp.NEG, operand=VariableNode(name="x"))
    """
    op: UnaryOp = UnaryOp.NEG
    operand: Optional[IRNode] = None
    
    def __post_init__(self):
        # Type follows operand for arithmetic, bool for logical
        if self.operand:
            if self.op == UnaryOp.NOT:
                self.dtype = IRType(IRTypeKind.Bool)
            else:
                self.dtype = self.operand.dtype
            self.rank = self.operand.rank
            self.is_jagged = self.operand.is_jagged
    
    def children(self) -> List[IRNode]:
        return [self.operand] if self.operand else []
    
    def __repr__(self) -> str:
        return f"UnaryOpNode({self.op.name}, operand={self.operand!r})"


@dataclass
class BinaryOpNode(IRNode):
    """
    Binary operation (arithmetic, comparison, logical, bitwise).
    
    Attributes:
        op: The binary operator
        left: Left operand
        right: Right operand
        broadcast_info: Information about rank broadcasting
        
    Example:
        >>> BinaryOpNode(
        ...     op=BinaryOp.ADD,
        ...     left=VariableNode(name="x"),
        ...     right=ConstantNode(value=1)
        ... )
    """
    op: BinaryOp = BinaryOp.ADD
    left: Optional[IRNode] = None
    right: Optional[IRNode] = None
    broadcast_info: Optional[BroadcastInfo] = None
    
    def children(self) -> List[IRNode]:
        result = []
        if self.left:
            result.append(self.left)
        if self.right:
            result.append(self.right)
        return result
    
    def __repr__(self) -> str:
        return f"BinaryOpNode({self.op.name}, left={self.left!r}, right={self.right!r})"


@dataclass
class TernaryOpNode(IRNode):
    """
    Conditional expression: if_true if condition else if_false.
    
    In C++: condition ? if_true : if_false
    
    Attributes:
        condition: Boolean condition expression
        if_true: Value when condition is true
        if_false: Value when condition is false
        
    Example:
        >>> TernaryOpNode(
        ...     condition=BinaryOpNode(op=BinaryOp.GT, left=x, right=zero),
        ...     if_true=x,
        ...     if_false=UnaryOpNode(op=UnaryOp.NEG, operand=x)
        ... )
    """
    condition: Optional[IRNode] = None
    if_true: Optional[IRNode] = None
    if_false: Optional[IRNode] = None
    
    def children(self) -> List[IRNode]:
        result = []
        if self.condition:
            result.append(self.condition)
        if self.if_true:
            result.append(self.if_true)
        if self.if_false:
            result.append(self.if_false)
        return result
    
    def __repr__(self) -> str:
        return (f"TernaryOpNode(condition={self.condition!r}, "
                f"if_true={self.if_true!r}, if_false={self.if_false!r})")


# =============================================================================
# Function and Method Nodes
# =============================================================================

@dataclass
class CallNode(IRNode):
    """
    Global function call (sqrt, cos, TMath::Gaus, etc.).
    
    Attributes:
        func: Function name as written
        args: List of argument expressions
        namespace: C++ namespace (e.g., "TMath" for TMath::Gaus)
        cpp_name: Actual C++ function name (may differ from func)
        headers: Required headers for this function
        
    Examples:
        >>> CallNode(func="sqrt", args=[x])
        >>> CallNode(func="Gaus", namespace="TMath", args=[x, mu, sigma])
    """
    func: str = ""
    args: List[IRNode] = field(default_factory=list)
    namespace: Optional[str] = None
    cpp_name: Optional[str] = None
    headers: List[str] = field(default_factory=list)
    
    def children(self) -> List[IRNode]:
        return list(self.args)
    
    def __repr__(self) -> str:
        ns = f"{self.namespace}::" if self.namespace else ""
        return f"CallNode({ns}{self.func}, args={self.args!r})"
    
    def full_cpp_name(self) -> str:
        """Return fully qualified C++ function name."""
        name = self.cpp_name or self.func
        if self.namespace:
            return f"{self.namespace}::{name}"
        return name


@dataclass
class MethodCallNode(IRNode):
    """
    Method call on a C++ object (track.getX(), particle.Px()).
    
    Attributes:
        object: The object expression
        method_name: Name of the method
        args: Method arguments
        class_name: C++ class name (from reflection)
        return_type: C++ return type (from reflection)
        signature: Full method signature (for debugging)
        
    Example:
        >>> MethodCallNode(
        ...     object=VariableNode(name="track"),
        ...     method_name="getX",
        ...     class_name="o2::tpc::TrackTPC"
        ... )
    """
    object: Optional[IRNode] = None
    method_name: str = ""
    args: List[IRNode] = field(default_factory=list)
    class_name: Optional[str] = None
    return_type: Optional[str] = None
    signature: Optional[str] = None
    
    def children(self) -> List[IRNode]:
        result = [self.object] if self.object else []
        result.extend(self.args)
        return result
    
    def __repr__(self) -> str:
        return (f"MethodCallNode(object={self.object!r}, "
                f"method={self.method_name!r}, args={self.args!r})")


@dataclass
class PropertyAccessNode(IRNode):
    """
    Property/field access on a C++ object (track.mPx, particle.fPdgCode).
    
    Attributes:
        object: The object expression
        property_name: Name of the property/field
        class_name: C++ class name (from reflection)
        property_type: C++ type of the property (from reflection)
        
    Example:
        >>> PropertyAccessNode(
        ...     object=VariableNode(name="particle"),
        ...     property_name="fPdgCode",
        ...     class_name="TParticle"
        ... )
    """
    object: Optional[IRNode] = None
    property_name: str = ""
    class_name: Optional[str] = None
    property_type: Optional[str] = None
    
    def children(self) -> List[IRNode]:
        return [self.object] if self.object else []
    
    def __repr__(self) -> str:
        return (f"PropertyAccessNode(object={self.object!r}, "
                f"property={self.property_name!r})")


# =============================================================================
# Broadcasting Nodes (Phase 8)
# =============================================================================

@dataclass
class MethodBroadcastNode(IRNode):
    """
    Element-wise method call on RVec<Object>.
    
    This node represents broadcasting a method call across all elements
    of an RVec containing objects. The result is an RVec of the method's
    return values.
    
    Attributes:
        target: The RVec<Object> expression being iterated
        method_name: Name of the method to call on each element
        element_type: C++ type of elements (e.g., "TLorentzVector")
        result_element_type: C++ return type of the method (e.g., "double")
        
    Example:
        # tracks.Pt() where tracks is RVec<TLorentzVector>
        >>> MethodBroadcastNode(
        ...     target=VariableNode(name="tracks"),
        ...     method_name="Pt",
        ...     element_type="TLorentzVector",
        ...     result_element_type="double"
        ... )
        # Result type: RVec<double>, rank=1
        
    Code Generation Pattern:
        [&]() -> ROOT::RVec<double> {
            ROOT::RVec<double> result;
            result.reserve(tracks.size());
            for (const auto& elem : tracks) {
                result.push_back(elem.Pt());
            }
            return result;
        }()
    """
    target: Optional[IRNode] = None
    method_name: str = ""
    element_type: str = ""
    result_element_type: str = ""
    
    def children(self) -> List[IRNode]:
        return [self.target] if self.target else []
    
    def __repr__(self) -> str:
        return (f"MethodBroadcastNode(target={self.target!r}, "
                f"method={self.method_name!r}, "
                f"element_type={self.element_type!r}, "
                f"result_type={self.result_element_type!r})")


@dataclass
class PropertyBroadcastNode(IRNode):
    """
    Element-wise property/field access on RVec<Object>.
    
    This node represents broadcasting a property access across all elements
    of an RVec containing objects. The result is an RVec of the property values.
    
    Attributes:
        target: The RVec<Object> expression being iterated
        property_name: Name of the property/field to access on each element
        element_type: C++ type of elements (e.g., "TParticle")
        result_element_type: C++ type of the property (e.g., "double")
        access_mode: How to access the property ("direct" or "reflection")
        
    Example:
        # particles.fPx where particles is RVec<TParticle>
        >>> PropertyBroadcastNode(
        ...     target=VariableNode(name="particles"),
        ...     property_name="fPx",
        ...     element_type="TParticle",
        ...     result_element_type="double",
        ...     access_mode="direct"
        ... )
        # Result type: RVec<double>, rank=1
        
    Code Generation Pattern:
        [&]() -> ROOT::RVec<double> {
            ROOT::RVec<double> result;
            result.reserve(particles.size());
            for (const auto& elem : particles) {
                result.push_back(elem.fPx);
            }
            return result;
        }()
    """
    target: Optional[IRNode] = None
    property_name: str = ""
    element_type: str = ""
    result_element_type: str = ""
    access_mode: str = "direct"  # "direct" or "reflection"
    
    def children(self) -> List[IRNode]:
        return [self.target] if self.target else []
    
    def __repr__(self) -> str:
        return (f"PropertyBroadcastNode(target={self.target!r}, "
                f"property={self.property_name!r}, "
                f"element_type={self.element_type!r}, "
                f"result_type={self.result_element_type!r})")


# =============================================================================
# Indexing Nodes
# =============================================================================

@dataclass
class SliceNode(IRNode):
    """
    Slice parameters (start:stop:step).
    
    Any of start, stop, step can be None (meaning default).
    
    Attributes:
        start: Start index (None = beginning)
        stop: Stop index (None = end)
        step: Step size (None = 1)
        
    Examples:
        >>> SliceNode(start=ConstantNode(1), stop=ConstantNode(10))  # [1:10]
        >>> SliceNode(step=ConstantNode(2))  # [::2]
        >>> SliceNode()  # [:]
    """
    start: Optional[IRNode] = None
    stop: Optional[IRNode] = None
    step: Optional[IRNode] = None
    
    def __post_init__(self):
        # Slices don't have a meaningful type themselves
        self.dtype = IRType(IRTypeKind.Unknown)
        self.rank = 0
    
    def children(self) -> List[IRNode]:
        result = []
        if self.start:
            result.append(self.start)
        if self.stop:
            result.append(self.stop)
        if self.step:
            result.append(self.step)
        return result
    
    def is_full_slice(self) -> bool:
        """Check if this is a full slice [:]."""
        return self.start is None and self.stop is None and self.step is None
    
    def has_step(self) -> bool:
        """Check if slice has a non-default step."""
        return self.step is not None
    
    def __repr__(self) -> str:
        return f"SliceNode(start={self.start!r}, stop={self.stop!r}, step={self.step!r})"


@dataclass
class SubscriptNode(IRNode):
    """
    Indexing or slicing operation (arr[i], arr[1:10], arr[:, 0]).
    
    Attributes:
        value: The array/collection being indexed
        indices: List of index expressions or SliceNodes
        is_fancy: Whether this is fancy indexing (index array)
        is_boolean_mask: Whether this is boolean mask indexing
        
    Examples:
        >>> SubscriptNode(value=arr, indices=[ConstantNode(5)])  # arr[5]
        >>> SubscriptNode(value=arr, indices=[SliceNode()])  # arr[:]
        >>> SubscriptNode(value=arr, indices=[SliceNode(), ConstantNode(0)])  # arr[:, 0]
    """
    value: Optional[IRNode] = None
    indices: List[Union[IRNode, SliceNode]] = field(default_factory=list)
    is_fancy: bool = False
    is_boolean_mask: bool = False
    
    def children(self) -> List[IRNode]:
        result = [self.value] if self.value else []
        for idx in self.indices:
            if isinstance(idx, IRNode):
                result.append(idx)
        return result
    
    def is_scalar_index(self) -> bool:
        """Check if this is simple scalar indexing (single integer index)."""
        if len(self.indices) != 1:
            return False
        idx = self.indices[0]
        if isinstance(idx, SliceNode):
            return False
        return idx.rank == 0 if isinstance(idx, IRNode) else False
    
    def is_slice(self) -> bool:
        """Check if any index is a slice."""
        return any(isinstance(idx, SliceNode) for idx in self.indices)
    
    def slice_dimensions(self) -> int:
        """Count number of slice dimensions."""
        return sum(1 for idx in self.indices if isinstance(idx, SliceNode))
    
    def __repr__(self) -> str:
        return f"SubscriptNode(value={self.value!r}, indices={self.indices!r})"


@dataclass
class CollectionIndexNode(IRNode):
    """
    Cross-collection indexing with safety (collision[track.GetCollisionIndex()]).
    
    This node represents accessing one collection using an index computed
    from another collection. It requires bounds checking for safety.
    
    Attributes:
        collection: The collection being indexed into
        index_expr: Expression computing the index
        safe_mode: Whether to generate bounds checking (default True)
        default_value: Value to return for out-of-bounds (default NaN)
        
    Example:
        >>> CollectionIndexNode(
        ...     collection=VariableNode(name="collision"),
        ...     index_expr=MethodCallNode(
        ...         object=VariableNode(name="track"),
        ...         method_name="GetCollisionIndex"
        ...     ),
        ...     safe_mode=True
        ... )
    """
    collection: Optional[IRNode] = None
    index_expr: Optional[IRNode] = None
    safe_mode: bool = True
    default_value: Optional[Any] = None
    
    def children(self) -> List[IRNode]:
        result = []
        if self.collection:
            result.append(self.collection)
        if self.index_expr:
            result.append(self.index_expr)
        return result
    
    def __repr__(self) -> str:
        return (f"CollectionIndexNode(collection={self.collection!r}, "
                f"index_expr={self.index_expr!r}, safe_mode={self.safe_mode})")


@dataclass
class RVecSliceNode(IRNode):
    """
    Represents a slice operation on RVec.
    
    This node represents Python-like slicing operations (pt[:3], pt[-3:], 
    pt[::2], etc.) and boolean masking (pt[pt > 1.0]).
    
    Attributes:
        target: The RVec being sliced
        start: Start index (None if open start)
        stop: Stop index (None if open end)
        step: Step value (None means step=1)
        slice_kind: Classification for code generation
        
    Examples:
        >>> RVecSliceNode(
        ...     target=VariableNode(name="pt"),
        ...     stop=ConstantNode(3),
        ...     slice_kind=SliceKind.FIRST_N
        ... )  # pt[:3]
        
        >>> RVecSliceNode(
        ...     target=VariableNode(name="pt"),
        ...     start=BinaryOpNode(...),  # pt > 1.0 expression
        ...     slice_kind=SliceKind.BOOLEAN
        ... )  # pt[pt > 1.0]
    """
    target: Optional[IRNode] = None
    start: Optional[IRNode] = None  # Also used for mask in BOOLEAN kind
    stop: Optional[IRNode] = None
    step: Optional[IRNode] = None
    slice_kind: SliceKind = SliceKind.FIRST_N
    
    def __post_init__(self):
        # Slicing always returns RVec (rank=1)
        self.rank = 1
    
    def children(self) -> List[IRNode]:
        result = []
        if self.target:
            result.append(self.target)
        if self.start:
            result.append(self.start)
        if self.stop:
            result.append(self.stop)
        if self.step:
            result.append(self.step)
        return result
    
    def __repr__(self) -> str:
        return (f"RVecSliceNode(target={self.target!r}, "
                f"start={self.start!r}, stop={self.stop!r}, "
                f"step={self.step!r}, kind={self.slice_kind})")


# =============================================================================
# Node Factory Functions
# =============================================================================

def make_constant(value: Union[int, float, bool]) -> ConstantNode:
    """Create a constant node from a Python value."""
    return ConstantNode(value=value)


def make_variable(name: str, namespace: Optional[str] = None) -> VariableNode:
    """Create a variable node."""
    return VariableNode(name=name, namespace=namespace)


def make_binary_op(op: BinaryOp, left: IRNode, right: IRNode) -> BinaryOpNode:
    """Create a binary operation node."""
    return BinaryOpNode(op=op, left=left, right=right)


def make_unary_op(op: UnaryOp, operand: IRNode) -> UnaryOpNode:
    """Create a unary operation node."""
    return UnaryOpNode(op=op, operand=operand)


def make_call(func: str, args: List[IRNode], namespace: Optional[str] = None) -> CallNode:
    """Create a function call node."""
    return CallNode(func=func, args=args, namespace=namespace)


def make_method_call(obj: IRNode, method: str, args: List[IRNode] = None) -> MethodCallNode:
    """Create a method call node."""
    return MethodCallNode(object=obj, method_name=method, args=args or [])


def make_subscript(value: IRNode, indices: List[Union[IRNode, SliceNode]]) -> SubscriptNode:
    """Create a subscript node."""
    return SubscriptNode(value=value, indices=indices)


def make_rvec_slice(target: IRNode, slice_kind: SliceKind,
                    start: Optional[IRNode] = None,
                    stop: Optional[IRNode] = None,
                    step: Optional[IRNode] = None) -> RVecSliceNode:
    """Create an RVec slice node."""
    return RVecSliceNode(
        target=target,
        start=start,
        stop=stop,
        step=step,
        slice_kind=slice_kind,
        dtype=target.dtype,  # Preserve element type
        rank=1  # Always returns RVec
    )
