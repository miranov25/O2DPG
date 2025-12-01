"""
PyArrow compute expression mapper for AliasDataFrame.

Phase 9a: Foundation for PyArrow acceleration.

Compiles Python/NumPy expressions to PyArrow compute function chains.
Provides significant speedup by eliminating Python dispatch overhead.

Example:
    # Current (NumPy) - Python dispatch per operation
    result = np.sqrt(px**2 + py**2)  # 3 Python calls, 3 temp arrays
    
    # Phase 9 (Arrow) - single C++ execution
    compiled = ArrowComputeMapper.compile("sqrt(px**2 + py**2)")
    result = compiled({'px': px_arrow, 'py': py_arrow})

Author: Claude (Coder)
Date: 2025-12-01
Version: 0.1.1 (Phase 9a - integer division fix)
"""

import ast
import warnings
from typing import Callable, Dict, Any, Optional

# Optional PyArrow import (follows Numba pattern)
try:
    import pyarrow as pa
    import pyarrow.compute as pc
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False
    pa = None
    pc = None


def _is_integer_type(arrow_type) -> bool:
    """
    Check if a PyArrow type is an integer type.
    
    Parameters
    ----------
    arrow_type : pa.DataType
        PyArrow data type to check
        
    Returns
    -------
    bool
        True if integer type (int8, int16, int32, int64, uint8, etc.)
    """
    if arrow_type is None:
        return False
    return pa.types.is_integer(arrow_type)


def _cast_to_float64_if_integer(arr):
    """
    Cast Arrow array to float64 if it's an integer type.
    
    This ensures division returns float results (Python 3 behavior).
    
    Parameters
    ----------
    arr : pa.Array or pa.Scalar
        Input array or scalar
        
    Returns
    -------
    pa.Array or pa.Scalar
        Original array if float, cast to float64 if integer
    """
    if isinstance(arr, pa.Array):
        if _is_integer_type(arr.type):
            return pc.cast(arr, pa.float64())
    elif isinstance(arr, pa.Scalar):
        if _is_integer_type(arr.type):
            return pa.scalar(float(arr.as_py()))
    return arr


class ArrowComputeMapper:
    """
    Maps NumPy/Python expressions to PyArrow compute operations.
    
    Uses AST parsing to transform expressions into PyArrow compute
    function chains. Falls back gracefully for unsupported expressions.
    
    Attributes
    ----------
    FUNC_MAP : dict
        Mapping from NumPy/AliasDataFrame function names to PyArrow names
    OP_MAP : dict
        Mapping from AST binary operator types to PyArrow function names
    UNARY_OP_MAP : dict
        Mapping from AST unary operator types to PyArrow function names
    CMP_MAP : dict
        Mapping from AST comparison types to PyArrow function names
        
    Examples
    --------
    >>> compiled = ArrowComputeMapper.compile("sqrt(px**2 + py**2)")
    >>> ctx = {'px': pa.array([3.0, 4.0]), 'py': pa.array([4.0, 3.0])}
    >>> result = compiled(ctx)
    >>> result.to_pylist()
    [5.0, 5.0]
    """
    
    # Function name mapping: AliasDataFrame/NumPy name → PyArrow name
    FUNC_MAP = {
        # Trigonometric
        'sin': 'sin', 
        'cos': 'cos', 
        'tan': 'tan',
        'arcsin': 'asin', 
        'arccos': 'acos', 
        'arctan': 'atan',
        'asin': 'asin', 
        'acos': 'acos', 
        'atan': 'atan',
        'arctan2': 'atan2', 
        'atan2': 'atan2',
        
        # Hyperbolic
        'sinh': 'sinh', 
        'cosh': 'cosh', 
        'tanh': 'tanh',
        'arcsinh': 'asinh', 
        'arccosh': 'acosh', 
        'arctanh': 'atanh',
        'asinh': 'asinh', 
        'acosh': 'acosh', 
        'atanh': 'atanh',
        
        # Exponential/Log - NOTE: log → ln in PyArrow!
        'exp': 'exp',
        'expm1': 'expm1',
        'log': 'ln',       # CRITICAL: PyArrow uses 'ln' for natural log
        'log10': 'log10',
        'log2': 'log2',
        'log1p': 'log1p',
        'sqrt': 'sqrt',
        'power': 'power',
        'pow': 'power',
        
        # Rounding
        'round': 'round', 
        'floor': 'floor', 
        'ceil': 'ceil',
        'trunc': 'trunc', 
        'abs': 'abs', 
        'sign': 'sign',
        
        # Additional math
        'isnan': 'is_nan',
        'isinf': 'is_inf',
        'isfinite': 'is_finite',
    }
    
    # Binary operator mapping: AST node type → PyArrow function name
    OP_MAP = {
        ast.Add: 'add',
        ast.Sub: 'subtract',
        ast.Mult: 'multiply',
        ast.Div: 'divide',
        ast.Pow: 'power',
        ast.Mod: 'mod',
        ast.FloorDiv: '__floordiv__',  # Special handling required
    }
    
    # Unary operator mapping
    UNARY_OP_MAP = {
        ast.USub: 'negate',
        ast.UAdd: None,  # No-op (unary plus)
        ast.Not: 'invert',
    }
    
    # Comparison operator mapping
    CMP_MAP = {
        ast.Eq: 'equal',
        ast.NotEq: 'not_equal',
        ast.Lt: 'less',
        ast.LtE: 'less_equal',
        ast.Gt: 'greater',
        ast.GtE: 'greater_equal',
    }
    
    # Cache for compiled expressions (class-level)
    _compile_cache: Dict[str, Optional[Callable]] = {}
    
    @classmethod
    def compile(cls, expr_str: str) -> Callable[[Dict[str, Any]], Any]:
        """
        Compile expression string to Arrow compute function.
        
        Parameters
        ----------
        expr_str : str
            Expression like "sqrt(px**2 + py**2)"
            
        Returns
        -------
        callable
            Function that takes context dict and returns Arrow array.
            Signature: (ctx: Dict[str, pa.Array]) -> pa.Array
            
        Raises
        ------
        ValueError
            If expression contains unsupported operations
        RuntimeError
            If PyArrow is not available
            
        Examples
        --------
        >>> compiled = ArrowComputeMapper.compile("sqrt(x**2 + y**2)")
        >>> ctx = {'x': pa.array([3.0]), 'y': pa.array([4.0])}
        >>> compiled(ctx).to_pylist()
        [5.0]
        """
        if not PYARROW_AVAILABLE:
            raise RuntimeError(
                "PyArrow is required for Arrow compute. "
                "Install with: pip install pyarrow>=14.0.0"
            )
        
        # Check cache
        if expr_str in cls._compile_cache:
            cached = cls._compile_cache[expr_str]
            if cached is None:
                raise ValueError(f"Expression previously failed to compile: {expr_str}")
            return cached
        
        try:
            tree = ast.parse(expr_str, mode='eval')
            compiled_fn = cls._compile_node(tree.body)
            cls._compile_cache[expr_str] = compiled_fn
            return compiled_fn
        except SyntaxError as e:
            cls._compile_cache[expr_str] = None
            raise ValueError(f"Invalid expression syntax: {expr_str}") from e
        except ValueError:
            cls._compile_cache[expr_str] = None
            raise
    
    @classmethod
    def _compile_node(cls, node: ast.AST) -> Callable[[Dict[str, Any]], Any]:
        """
        Recursively compile AST node to Arrow compute callable.
        
        Parameters
        ----------
        node : ast.AST
            AST node to compile
            
        Returns
        -------
        callable
            Function that evaluates node given context
        """
        if isinstance(node, ast.BinOp):
            return cls._compile_binop(node)
        
        elif isinstance(node, ast.UnaryOp):
            return cls._compile_unaryop(node)
        
        elif isinstance(node, ast.Call):
            return cls._compile_call(node)
        
        elif isinstance(node, ast.Name):
            return cls._compile_name(node)
        
        elif isinstance(node, ast.Constant):
            return cls._compile_constant(node)
        
        elif isinstance(node, ast.Num):
            # Python 3.7 compatibility (ast.Num deprecated in 3.8+)
            return cls._compile_num(node)
        
        elif isinstance(node, ast.Attribute):
            return cls._compile_attribute(node)
        
        elif isinstance(node, ast.Compare):
            return cls._compile_compare(node)
        
        elif isinstance(node, ast.IfExp):
            return cls._compile_ifexp(node)
        
        elif isinstance(node, ast.Subscript):
            return cls._compile_subscript(node)
        
        else:
            raise ValueError(f"Unsupported AST node type: {type(node).__name__}")
    
    @classmethod
    def _compile_binop(cls, node: ast.BinOp) -> Callable:
        """Compile binary operation (e.g., x + y, x * y)."""
        left_fn = cls._compile_node(node.left)
        right_fn = cls._compile_node(node.right)
        op_type = type(node.op)
        
        # Special case: floor division (a // b)
        if op_type == ast.FloorDiv:
            def floordiv_fn(ctx):
                left_val = _cast_to_float64_if_integer(left_fn(ctx))
                right_val = _cast_to_float64_if_integer(right_fn(ctx))
                return pc.floor(pc.divide(left_val, right_val))
            return floordiv_fn
        
        # Special case: true division (a / b) - cast integers to float
        # This matches Python 3 behavior where `/` always returns float
        if op_type == ast.Div:
            def truediv_fn(ctx):
                left_val = _cast_to_float64_if_integer(left_fn(ctx))
                right_val = _cast_to_float64_if_integer(right_fn(ctx))
                return pc.divide(left_val, right_val)
            return truediv_fn
        
        op_name = cls.OP_MAP.get(op_type)
        if op_name is None:
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        
        op_func = getattr(pc, op_name)
        
        def binop_fn(ctx):
            left_val = left_fn(ctx)
            right_val = right_fn(ctx)
            return op_func(left_val, right_val)
        
        return binop_fn
    
    @classmethod
    def _compile_unaryop(cls, node: ast.UnaryOp) -> Callable:
        """Compile unary operation (e.g., -x, not x)."""
        operand_fn = cls._compile_node(node.operand)
        op_name = cls.UNARY_OP_MAP.get(type(node.op))
        
        if op_name is None:
            # UAdd (+x) is a no-op
            return operand_fn
        
        op_func = getattr(pc, op_name)
        
        def unaryop_fn(ctx):
            return op_func(operand_fn(ctx))
        
        return unaryop_fn
    
    @classmethod
    def _compile_call(cls, node: ast.Call) -> Callable:
        """Compile function call (e.g., sqrt(x), sin(y))."""
        # Get function name
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Handle np.sqrt, math.sin, etc.
            func_name = node.func.attr
        else:
            raise ValueError(f"Unsupported function call syntax: {ast.dump(node)}")
        
        # Special case: clip(x, min, max)
        if func_name == 'clip':
            return cls._compile_clip(node)
        
        # Special case: where(condition, x, y)
        if func_name == 'where':
            return cls._compile_where(node)
        
        # Map to Arrow function
        arrow_name = cls.FUNC_MAP.get(func_name)
        if arrow_name is None:
            available = ', '.join(sorted(cls.FUNC_MAP.keys()))
            raise ValueError(
                f"No PyArrow equivalent for function '{func_name}'. "
                f"Available functions: {available}"
            )
        
        arrow_func = getattr(pc, arrow_name, None)
        if arrow_func is None:
            raise ValueError(f"PyArrow function '{arrow_name}' not found in pyarrow.compute")
        
        # Compile arguments
        arg_fns = [cls._compile_node(arg) for arg in node.args]
        
        def call_fn(ctx):
            args = [fn(ctx) for fn in arg_fns]
            return arrow_func(*args)
        
        return call_fn
    
    @classmethod
    def _compile_clip(cls, node: ast.Call) -> Callable:
        """Compile clip(x, min, max) using Arrow primitives."""
        if len(node.args) != 3:
            raise ValueError(
                f"clip() requires exactly 3 arguments (x, min, max), "
                f"got {len(node.args)}"
            )
        
        x_fn = cls._compile_node(node.args[0])
        min_fn = cls._compile_node(node.args[1])
        max_fn = cls._compile_node(node.args[2])
        
        def clip_fn(ctx):
            x = x_fn(ctx)
            min_val = min_fn(ctx)
            max_val = max_fn(ctx)
            # clip(x, min, max) = min(max(x, min), max)
            result = pc.max_element_wise(x, min_val)
            result = pc.min_element_wise(result, max_val)
            return result
        
        return clip_fn
    
    @classmethod
    def _compile_where(cls, node: ast.Call) -> Callable:
        """Compile where(condition, x, y) using Arrow if_else."""
        if len(node.args) != 3:
            raise ValueError(
                f"where() requires exactly 3 arguments (condition, x, y), "
                f"got {len(node.args)}"
            )
        
        cond_fn = cls._compile_node(node.args[0])
        x_fn = cls._compile_node(node.args[1])
        y_fn = cls._compile_node(node.args[2])
        
        def where_fn(ctx):
            condition = cond_fn(ctx)
            x = x_fn(ctx)
            y = y_fn(ctx)
            return pc.if_else(condition, x, y)
        
        return where_fn
    
    @classmethod
    def _compile_name(cls, node: ast.Name) -> Callable:
        """Compile variable reference (e.g., x, py)."""
        name = node.id
        
        def name_fn(ctx):
            if name not in ctx:
                raise KeyError(
                    f"Variable '{name}' not found in context. "
                    f"Available: {list(ctx.keys())}"
                )
            value = ctx[name]
            # Convert to Arrow if needed
            if isinstance(value, (pa.Array, pa.ChunkedArray, pa.Scalar)):
                return value
            elif hasattr(value, '__array__'):
                # numpy array
                return pa.array(value)
            elif hasattr(value, 'values'):
                # pandas Series
                return pa.array(value.values)
            else:
                return pa.array(value)
        
        return name_fn
    
    @classmethod
    def _compile_constant(cls, node: ast.Constant) -> Callable:
        """Compile constant value (e.g., 2, 3.14, 40.)."""
        value = node.value
        
        def constant_fn(ctx):
            return pa.scalar(value)
        
        return constant_fn
    
    @classmethod
    def _compile_num(cls, node: ast.Num) -> Callable:
        """Compile numeric constant (Python 3.7 compatibility)."""
        value = node.n
        
        def num_fn(ctx):
            return pa.scalar(value)
        
        return num_fn
    
    @classmethod
    def _compile_attribute(cls, node: ast.Attribute) -> Callable:
        """
        Compile attribute access (e.g., subframe.column, np.pi).
        
        Supports:
        - subframe.column_name → ctx['subframe.column_name'] or ctx['subframe']['column_name']
        - np.pi, math.e → constant values
        """
        import math
        
        # Handle common constants
        if isinstance(node.value, ast.Name):
            obj_name = node.value.id
            attr_name = node.attr
            
            # numpy/math constants
            if obj_name in ('np', 'numpy') and attr_name == 'pi':
                return lambda ctx: pa.scalar(math.pi)
            if obj_name in ('np', 'numpy') and attr_name == 'e':
                return lambda ctx: pa.scalar(math.e)
            if obj_name == 'math' and attr_name == 'pi':
                return lambda ctx: pa.scalar(math.pi)
            if obj_name == 'math' and attr_name == 'e':
                return lambda ctx: pa.scalar(math.e)
            
            # Subframe column reference: subframe.column
            full_name = f"{obj_name}.{attr_name}"
            
            def attr_fn(ctx):
                # Try full dotted name first
                if full_name in ctx:
                    value = ctx[full_name]
                    if isinstance(value, pa.Array):
                        return value
                    return pa.array(value)
                
                # Try nested dict access
                if obj_name in ctx:
                    obj = ctx[obj_name]
                    if isinstance(obj, dict) and attr_name in obj:
                        value = obj[attr_name]
                        if isinstance(value, pa.Array):
                            return value
                        return pa.array(value)
                
                raise KeyError(
                    f"Cannot resolve '{full_name}'. "
                    f"Context keys: {list(ctx.keys())}"
                )
            
            return attr_fn
        
        raise ValueError(f"Unsupported attribute access pattern: {ast.dump(node)}")
    
    @classmethod
    def _compile_compare(cls, node: ast.Compare) -> Callable:
        """Compile comparison (e.g., x > 0, a == b)."""
        # Only support single comparisons (not chained like a < b < c)
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise ValueError(
                "Chained comparisons not supported (e.g., a < b < c). "
                "Use explicit 'and' expressions instead."
            )
        
        left_fn = cls._compile_node(node.left)
        right_fn = cls._compile_node(node.comparators[0])
        
        cmp_name = cls.CMP_MAP.get(type(node.ops[0]))
        if cmp_name is None:
            raise ValueError(f"Unsupported comparison operator: {type(node.ops[0]).__name__}")
        
        cmp_func = getattr(pc, cmp_name)
        
        def compare_fn(ctx):
            return cmp_func(left_fn(ctx), right_fn(ctx))
        
        return compare_fn
    
    @classmethod
    def _compile_ifexp(cls, node: ast.IfExp) -> Callable:
        """Compile conditional expression (e.g., x if condition else y)."""
        test_fn = cls._compile_node(node.test)
        body_fn = cls._compile_node(node.body)
        orelse_fn = cls._compile_node(node.orelse)
        
        def ifexp_fn(ctx):
            condition = test_fn(ctx)
            true_val = body_fn(ctx)
            false_val = orelse_fn(ctx)
            return pc.if_else(condition, true_val, false_val)
        
        return ifexp_fn
    
    @classmethod
    def _compile_subscript(cls, node: ast.Subscript) -> Callable:
        """Compile subscript access (e.g., arr[0], data['col'])."""
        # This is a simplified implementation for common cases
        raise ValueError(
            f"Subscript expressions not yet supported: {ast.dump(node)}. "
            "Use explicit variable names instead."
        )
    
    @classmethod
    def is_supported(cls, expr_str: str) -> bool:
        """
        Check if expression can be compiled to Arrow.
        
        Parameters
        ----------
        expr_str : str
            Expression to check
            
        Returns
        -------
        bool
            True if expression can be compiled, False otherwise
            
        Examples
        --------
        >>> ArrowComputeMapper.is_supported("sqrt(x**2 + y**2)")
        True
        >>> ArrowComputeMapper.is_supported("unknown_func(x)")
        False
        """
        if not PYARROW_AVAILABLE:
            return False
        
        try:
            cls.compile(expr_str)
            return True
        except (ValueError, SyntaxError):
            return False
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the expression compilation cache."""
        cls._compile_cache.clear()
    
    @classmethod
    def get_supported_functions(cls) -> list:
        """
        Get list of supported function names.
        
        Returns
        -------
        list
            Sorted list of function names that can be used in expressions
        """
        return sorted(cls.FUNC_MAP.keys())


def evaluate_expression_arrow(
    expr_str: str,
    context: Dict[str, Any],
    fallback_fn: Optional[Callable] = None
) -> Any:
    """
    Evaluate expression using PyArrow compute, with optional fallback.
    
    This is a convenience function that handles Arrow/NumPy conversion
    and provides fallback for unsupported expressions.
    
    Parameters
    ----------
    expr_str : str
        Expression to evaluate
    context : dict
        Variable bindings (can be Arrow arrays, numpy arrays, or pandas Series)
    fallback_fn : callable, optional
        Function to call if Arrow compilation fails.
        Signature: (expr_str, context) -> result
        
    Returns
    -------
    pa.Array or numpy.ndarray
        Result of evaluation (Arrow array if Arrow succeeded, otherwise fallback result)
        
    Examples
    --------
    >>> import numpy as np
    >>> ctx = {'x': np.array([1.0, 4.0, 9.0])}
    >>> result = evaluate_expression_arrow("sqrt(x)", ctx)
    >>> result.to_numpy()
    array([1., 2., 3.])
    """
    if not PYARROW_AVAILABLE:
        if fallback_fn is not None:
            return fallback_fn(expr_str, context)
        raise RuntimeError("PyArrow not available and no fallback provided")
    
    try:
        compiled = ArrowComputeMapper.compile(expr_str)
        
        # Convert context values to Arrow arrays
        arrow_ctx = {}
        for name, value in context.items():
            if isinstance(value, (pa.Array, pa.ChunkedArray)):
                arrow_ctx[name] = value
            elif hasattr(value, 'values'):
                # pandas Series
                arrow_ctx[name] = pa.array(value.values)
            elif hasattr(value, '__array__'):
                # numpy array or array-like
                arrow_ctx[name] = pa.array(value)
            else:
                # Scalar or other
                arrow_ctx[name] = value
        
        return compiled(arrow_ctx)
        
    except (ValueError, SyntaxError) as e:
        if fallback_fn is not None:
            warnings.warn(
                f"Arrow compilation failed for '{expr_str}': {e}. "
                "Falling back to NumPy.",
                RuntimeWarning
            )
            return fallback_fn(expr_str, context)
        raise


# Module-level convenience
def is_arrow_available() -> bool:
    """Check if PyArrow is available."""
    return PYARROW_AVAILABLE
