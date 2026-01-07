"""
Phase 13.3.DSL D5+D6: C++ Code Generation for ROOT Linear Algebra Types

This module generates C++ code for TMatrixD/TVectorD operations:
- Element access with bounds checking
- Row/column extraction returning RVec<T>
- Submatrix extraction returning RVec<RVec<T>>
- Vector slicing with various patterns

All generated code uses lambda expressions for safe evaluation
and proper negative index normalization.
"""

from typing import Optional, Union, Tuple, List
from dataclasses import dataclass

from .ir_nodes_linalg import (
    LinalgAccessNode, LinalgSliceKind, SliceParams
)


__all__ = [
    'LinalgCodeGenerator',
    'generate_linalg_code',
]


@dataclass
class GeneratedCode:
    """Result of code generation."""
    code: str
    dependencies: List[str]
    result_type: str


class LinalgCodeGenerator:
    """
    Generates C++ code for TMatrixD/TVectorD operations.
    
    Features:
    - Bounds-checked access with NaN fallback for scalars
    - Empty RVec fallback for out-of-bounds slices
    - Negative index normalization: idx >= 0 ? idx : (N + idx)
    - Lambda-wrapped expressions for safe evaluation
    
    Phase 13.3.DSL D5+D6.
    """
    
    def __init__(self, safe_indexing: bool = True):
        """
        Initialize code generator.
        
        Args:
            safe_indexing: If True, generate bounds-checked code
        """
        self.safe_indexing = safe_indexing
        self._dependencies: List[str] = []
    
    def generate(self, node: LinalgAccessNode) -> GeneratedCode:
        """
        Generate C++ code for a linalg access node.
        
        Args:
            node: LinalgAccessNode describing the operation
            
        Returns:
            GeneratedCode with code string, dependencies, and result type
        """
        self._dependencies = []
        
        # Use node's safe_indexing setting if specified
        safe = node.safe_indexing if node.safe_indexing is not None else self.safe_indexing
        
        if node.access_kind == LinalgSliceKind.MATRIX_ELEMENT:
            code = self._gen_matrix_element(node, safe)
        elif node.access_kind == LinalgSliceKind.MATRIX_ROW:
            code = self._gen_matrix_row(node, safe)
        elif node.access_kind == LinalgSliceKind.MATRIX_COLUMN:
            code = self._gen_matrix_column(node, safe)
        elif node.access_kind == LinalgSliceKind.MATRIX_SUBMATRIX:
            code = self._gen_matrix_submatrix(node, safe)
        elif node.access_kind == LinalgSliceKind.VECTOR_ELEMENT:
            code = self._gen_vector_element(node, safe)
        elif node.access_kind == LinalgSliceKind.VECTOR_SLICE:
            code = self._gen_vector_slice(node, safe)
        else:
            raise ValueError(f"Unknown access kind: {node.access_kind}")
        
        return GeneratedCode(
            code=code,
            dependencies=list(self._dependencies),
            result_type=node.get_result_cpp_type()
        )
    
    # =========================================================================
    # Matrix Operations
    # =========================================================================
    
    def _gen_matrix_element(self, node: LinalgAccessNode, safe: bool) -> str:
        """
        Generate code for matrix element access: mat(i, j).
        
        Safe mode returns NaN for out-of-bounds access.
        """
        target = node.target
        row = self._format_index(node.row_index)
        col = self._format_index(node.col_index)
        elem_type = node.element_type
        
        if safe:
            # Lambda with bounds checking
            return f"""[&]() {{
    const auto& m = {target};
    int nrows = m.GetNrows();
    int ncols = m.GetNcols();
    int r = {row};
    int c = {col};
    int r_norm = r >= 0 ? r : nrows + r;
    int c_norm = c >= 0 ? c : ncols + c;
    if (r_norm < 0 || r_norm >= nrows || c_norm < 0 || c_norm >= ncols) {{
        return std::numeric_limits<{elem_type}>::quiet_NaN();
    }}
    return static_cast<{elem_type}>(m(r_norm, c_norm));
}}()"""
        else:
            # Direct access (unsafe)
            return f"{target}({row}, {col})"
    
    def _gen_matrix_row(self, node: LinalgAccessNode, safe: bool) -> str:
        """
        Generate code for matrix row extraction: mat[i] → RVec<T>.
        
        Returns empty RVec for out-of-bounds row.
        """
        target = node.target
        row = self._format_index(node.row_index)
        elem_type = node.element_type
        
        self._dependencies.append("ROOT/RVec.hxx")
        
        if safe:
            return f"""[&]() {{
    const auto& m = {target};
    int nrows = m.GetNrows();
    int ncols = m.GetNcols();
    int r = {row};
    int r_norm = r >= 0 ? r : nrows + r;
    if (r_norm < 0 || r_norm >= nrows) {{
        return ROOT::RVec<{elem_type}>();
    }}
    ROOT::RVec<{elem_type}> result(ncols);
    for (int j = 0; j < ncols; ++j) {{
        result[j] = m(r_norm, j);
    }}
    return result;
}}()"""
        else:
            return f"""[&]() {{
    const auto& m = {target};
    int ncols = m.GetNcols();
    ROOT::RVec<{elem_type}> result(ncols);
    for (int j = 0; j < ncols; ++j) {{
        result[j] = m({row}, j);
    }}
    return result;
}}()"""
    
    def _gen_matrix_column(self, node: LinalgAccessNode, safe: bool) -> str:
        """
        Generate code for matrix column extraction: mat[:, j] → RVec<T>.
        
        Returns empty RVec for out-of-bounds column.
        """
        target = node.target
        col = self._format_index(node.col_index)
        elem_type = node.element_type
        
        self._dependencies.append("ROOT/RVec.hxx")
        
        if safe:
            return f"""[&]() {{
    const auto& m = {target};
    int nrows = m.GetNrows();
    int ncols = m.GetNcols();
    int c = {col};
    int c_norm = c >= 0 ? c : ncols + c;
    if (c_norm < 0 || c_norm >= ncols) {{
        return ROOT::RVec<{elem_type}>();
    }}
    ROOT::RVec<{elem_type}> result(nrows);
    for (int i = 0; i < nrows; ++i) {{
        result[i] = m(i, c_norm);
    }}
    return result;
}}()"""
        else:
            return f"""[&]() {{
    const auto& m = {target};
    int nrows = m.GetNrows();
    ROOT::RVec<{elem_type}> result(nrows);
    for (int i = 0; i < nrows; ++i) {{
        result[i] = m(i, {col});
    }}
    return result;
}}()"""
    
    def _gen_matrix_submatrix(self, node: LinalgAccessNode, safe: bool) -> str:
        """
        Generate code for submatrix extraction: mat[a:b, c:d] → RVec<RVec<T>>.
        """
        target = node.target
        elem_type = node.element_type
        
        row_slice = node.row_index if isinstance(node.row_index, SliceParams) else SliceParams()
        col_slice = node.col_index if isinstance(node.col_index, SliceParams) else SliceParams()
        
        self._dependencies.append("ROOT/RVec.hxx")
        
        # Generate slice bounds
        row_start = self._format_slice_bound(row_slice.start, "0")
        row_stop = self._format_slice_bound(row_slice.stop, "nrows")
        col_start = self._format_slice_bound(col_slice.start, "0")
        col_stop = self._format_slice_bound(col_slice.stop, "ncols")
        
        return f"""[&]() {{
    const auto& m = {target};
    int nrows = m.GetNrows();
    int ncols = m.GetNcols();
    int r_start = {row_start};
    int r_stop = {row_stop};
    int c_start = {col_start};
    int c_stop = {col_stop};
    // Normalize negative indices
    r_start = r_start >= 0 ? r_start : nrows + r_start;
    r_stop = r_stop >= 0 ? r_stop : nrows + r_stop;
    c_start = c_start >= 0 ? c_start : ncols + c_start;
    c_stop = c_stop >= 0 ? c_stop : ncols + c_stop;
    // Clamp to valid range
    r_start = std::max(0, std::min(r_start, nrows));
    r_stop = std::max(0, std::min(r_stop, nrows));
    c_start = std::max(0, std::min(c_start, ncols));
    c_stop = std::max(0, std::min(c_stop, ncols));
    ROOT::RVec<ROOT::RVec<{elem_type}>> result;
    for (int i = r_start; i < r_stop; ++i) {{
        ROOT::RVec<{elem_type}> row(c_stop - c_start);
        for (int j = c_start; j < c_stop; ++j) {{
            row[j - c_start] = m(i, j);
        }}
        result.push_back(row);
    }}
    return result;
}}()"""
    
    # =========================================================================
    # Vector Operations
    # =========================================================================
    
    def _gen_vector_element(self, node: LinalgAccessNode, safe: bool) -> str:
        """
        Generate code for vector element access: vec[i].
        
        Safe mode returns NaN for out-of-bounds access.
        """
        target = node.target
        index = self._format_index(node.row_index)  # Vector uses row_index
        elem_type = node.element_type
        
        if safe:
            return f"""[&]() {{
    const auto& v = {target};
    int n = v.GetNrows();
    int i = {index};
    int i_norm = i >= 0 ? i : n + i;
    if (i_norm < 0 || i_norm >= n) {{
        return std::numeric_limits<{elem_type}>::quiet_NaN();
    }}
    return static_cast<{elem_type}>(v[i_norm]);
}}()"""
        else:
            return f"{target}[{index}]"
    
    def _gen_vector_slice(self, node: LinalgAccessNode, safe: bool) -> str:
        """
        Generate code for vector slicing: vec[:n], vec[::2], etc.
        
        Handles various slice patterns:
        - [:n]   → first n elements
        - [n:]   → from n to end
        - [a:b]  → range
        - [::k]  → every k-th element
        - [::-1] → reverse
        - [-n:]  → last n elements
        """
        target = node.target
        elem_type = node.element_type
        params = node.slice_params or SliceParams()
        
        self._dependencies.append("ROOT/RVec.hxx")
        
        # Dispatch to specific slice pattern
        if params.is_reverse():
            return self._gen_vector_reverse(target, elem_type)
        elif params.is_step_slice():
            return self._gen_vector_step_slice(target, params, elem_type)
        elif params.is_first_n():
            return self._gen_vector_first_n(target, params.stop, elem_type, safe)
        elif params.is_from_n():
            return self._gen_vector_from_index(target, params.start, elem_type, safe)
        elif params.is_range():
            return self._gen_vector_range(target, params.start, params.stop, elem_type, safe)
        else:
            # Full slice [:] - copy entire vector
            return self._gen_vector_full_copy(target, elem_type)
    
    def _gen_vector_first_n(self, target: str, stop: Union[int, str], 
                            elem_type: str, safe: bool) -> str:
        """Generate code for vec[:n] - first n elements."""
        stop_expr = self._format_index(stop)
        
        return f"""[&]() {{
    const auto& v = {target};
    int n = v.GetNrows();
    int stop = {stop_expr};
    stop = stop >= 0 ? stop : n + stop;
    stop = std::min(stop, n);
    stop = std::max(0, stop);
    ROOT::RVec<{elem_type}> result(stop);
    for (int i = 0; i < stop; ++i) {{
        result[i] = v[i];
    }}
    return result;
}}()"""
    
    def _gen_vector_from_index(self, target: str, start: Union[int, str],
                               elem_type: str, safe: bool) -> str:
        """Generate code for vec[n:] - from n to end."""
        start_expr = self._format_index(start)
        
        return f"""[&]() {{
    const auto& v = {target};
    int n = v.GetNrows();
    int start = {start_expr};
    start = start >= 0 ? start : n + start;
    start = std::max(0, std::min(start, n));
    ROOT::RVec<{elem_type}> result(n - start);
    for (int i = start; i < n; ++i) {{
        result[i - start] = v[i];
    }}
    return result;
}}()"""
    
    def _gen_vector_range(self, target: str, start: Union[int, str],
                          stop: Union[int, str], elem_type: str, safe: bool) -> str:
        """Generate code for vec[a:b] - range."""
        start_expr = self._format_index(start)
        stop_expr = self._format_index(stop)
        
        return f"""[&]() {{
    const auto& v = {target};
    int n = v.GetNrows();
    int start = {start_expr};
    int stop = {stop_expr};
    start = start >= 0 ? start : n + start;
    stop = stop >= 0 ? stop : n + stop;
    start = std::max(0, std::min(start, n));
    stop = std::max(0, std::min(stop, n));
    int len = std::max(0, stop - start);
    ROOT::RVec<{elem_type}> result(len);
    for (int i = 0; i < len; ++i) {{
        result[i] = v[start + i];
    }}
    return result;
}}()"""
    
    def _gen_vector_step_slice(self, target: str, params: SliceParams,
                               elem_type: str) -> str:
        """Generate code for vec[::k] - every k-th element."""
        start_expr = self._format_slice_bound(params.start, "0")
        stop_expr = self._format_slice_bound(params.stop, "n")
        step_expr = self._format_index(params.step)
        
        return f"""[&]() {{
    const auto& v = {target};
    int n = v.GetNrows();
    int start = {start_expr};
    int stop = {stop_expr};
    int step = {step_expr};
    start = start >= 0 ? start : n + start;
    stop = stop >= 0 ? stop : n + stop;
    start = std::max(0, std::min(start, n));
    stop = std::max(0, std::min(stop, n));
    if (step == 0) step = 1;  // Prevent infinite loop
    ROOT::RVec<{elem_type}> result;
    if (step > 0) {{
        for (int i = start; i < stop; i += step) {{
            result.push_back(v[i]);
        }}
    }} else {{
        // Negative step: iterate backwards
        for (int i = stop - 1; i >= start; i += step) {{
            result.push_back(v[i]);
        }}
    }}
    return result;
}}()"""
    
    def _gen_vector_reverse(self, target: str, elem_type: str) -> str:
        """Generate code for vec[::-1] - reverse."""
        return f"""[&]() {{
    const auto& v = {target};
    int n = v.GetNrows();
    ROOT::RVec<{elem_type}> result(n);
    for (int i = 0; i < n; ++i) {{
        result[i] = v[n - 1 - i];
    }}
    return result;
}}()"""
    
    def _gen_vector_full_copy(self, target: str, elem_type: str) -> str:
        """Generate code for vec[:] - full copy."""
        return f"""[&]() {{
    const auto& v = {target};
    int n = v.GetNrows();
    ROOT::RVec<{elem_type}> result(n);
    for (int i = 0; i < n; ++i) {{
        result[i] = v[i];
    }}
    return result;
}}()"""
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _format_index(self, index: Union[int, str, None]) -> str:
        """Format an index value for code generation."""
        if index is None:
            return "0"
        if isinstance(index, int):
            return str(index)
        return str(index)
    
    def _format_slice_bound(self, bound: Union[int, str, None], 
                            default: str) -> str:
        """Format a slice bound, using default if None."""
        if bound is None:
            return default
        if isinstance(bound, int):
            return str(bound)
        return str(bound)


# =============================================================================
# Module-Level Function
# =============================================================================

def generate_linalg_code(node: LinalgAccessNode, 
                         safe_indexing: bool = True) -> GeneratedCode:
    """
    Generate C++ code for a linear algebra access node.
    
    Convenience function that creates a generator and calls generate().
    
    Args:
        node: LinalgAccessNode describing the operation
        safe_indexing: Whether to use bounds-checked access
        
    Returns:
        GeneratedCode with code string, dependencies, and result type
    """
    generator = LinalgCodeGenerator(safe_indexing=safe_indexing)
    return generator.generate(node)
