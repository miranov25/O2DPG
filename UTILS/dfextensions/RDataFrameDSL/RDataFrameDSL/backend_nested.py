"""
Phase 13.3.DSL D7: C++ Code Generation for Nested RVec 2D Slicing

Generates C++ code for RVec<RVec<T>> operations:
- Column extraction (fail-closed): nested[:, j] → RVec<T>
- Column slicing (clamp per-row): nested[:, a:b] → RVec<RVec<T>>

Two-Tier Ragged Policy:
- Integer column index: FAIL-CLOSED - throws std::runtime_error if any row too short
- Slice column index: CLAMP PER-ROW - shorter rows produce shorter results
"""

from typing import Optional, Union, List
from dataclasses import dataclass

from .ir_nodes_nested import (
    NestedAccessNode, NestedSliceKind, SliceParams
)


__all__ = [
    'NestedCodeGenerator',
    'generate_nested_code',
]


@dataclass
class GeneratedCode:
    """Result of code generation."""
    code: str
    dependencies: List[str]
    result_type: str


class NestedCodeGenerator:
    """
    Generates C++ code for RVec<RVec<T>> (nested RVec) operations.
    
    Two-Tier Ragged Policy Implementation:
    - NESTED_COLUMN_EXTRACT: Fail-closed with std::runtime_error
    - NESTED_COLUMN_SLICE: Clamp per-row, variable-length results
    
    All generated code uses lambda expressions for safe evaluation.
    
    Phase 13.3.DSL D7.
    """
    
    def __init__(self):
        """Initialize code generator."""
        self._dependencies: List[str] = []
    
    def generate(self, node: NestedAccessNode) -> GeneratedCode:
        """
        Generate C++ code for a nested access node.
        
        Args:
            node: NestedAccessNode describing the operation
            
        Returns:
            GeneratedCode with code string, dependencies, and result type
        """
        self._dependencies = ["ROOT/RVec.hxx"]
        
        if node.access_kind == NestedSliceKind.NESTED_COLUMN_EXTRACT:
            code = self._gen_column_extract(node)
        elif node.access_kind == NestedSliceKind.NESTED_COLUMN_SLICE:
            code = self._gen_column_slice(node)
        elif node.access_kind == NestedSliceKind.NESTED_ELEMENT:
            code = self._gen_element_access(node)
        elif node.access_kind == NestedSliceKind.NESTED_ROW_ACCESS:
            code = self._gen_row_access(node)
        elif node.access_kind == NestedSliceKind.NESTED_ROW_SLICE:
            code = self._gen_row_slice(node)
        else:
            raise ValueError(f"Unknown nested access kind: {node.access_kind}")
        
        return GeneratedCode(
            code=code,
            dependencies=list(self._dependencies),
            result_type=node.get_result_cpp_type()
        )
    
    # =========================================================================
    # Column Operations (D7 Core)
    # =========================================================================
    
    def _gen_column_extract(self, node: NestedAccessNode) -> str:
        """
        Generate code for column extraction: nested[:, j] → RVec<T>.
        
        FAIL-CLOSED policy: throws std::runtime_error if ANY row is 
        missing the requested element.
        
        Supports negative indexing: -1 = last element, etc.
        """
        target = node.target
        col = self._format_index(node.col_index)
        elem_type = node.element_type
        
        self._dependencies.append("stdexcept")
        
        return f"""[&]() {{
    const auto& outer = {target};
    int j = {col};
    ROOT::RVec<{elem_type}> result;
    result.reserve(outer.size());
    for (size_t i = 0; i < outer.size(); ++i) {{
        const auto& row = outer[i];
        int row_size = static_cast<int>(row.size());
        int j_norm = j >= 0 ? j : row_size + j;
        if (j_norm < 0 || j_norm >= row_size) {{
            throw std::runtime_error(
                "DSL: Column index " + std::to_string(j) +
                " out of bounds at row " + std::to_string(i) +
                " (row size: " + std::to_string(row_size) + ")"
            );
        }}
        result.push_back(row[j_norm]);
    }}
    return result;
}}()"""
    
    def _gen_column_slice(self, node: NestedAccessNode) -> str:
        """
        Generate code for column slicing: nested[:, a:b] → RVec<RVec<T>>.
        
        CLAMP-PER-ROW policy: each row is sliced independently,
        clamping indices to valid range. Shorter rows produce shorter results.
        """
        target = node.target
        elem_type = node.element_type
        
        col_slice = node.col_index if isinstance(node.col_index, SliceParams) else SliceParams()
        
        # Handle different slice patterns
        if col_slice.is_step_slice():
            return self._gen_column_slice_with_step(target, col_slice, elem_type)
        else:
            return self._gen_column_slice_range(target, col_slice, elem_type)
    
    def _gen_column_slice_range(self, target: str, params: SliceParams, 
                                 elem_type: str) -> str:
        """Generate column slice for simple range [a:b]."""
        start_expr = self._format_slice_bound(params.start, "0")
        stop_expr = self._format_slice_bound(params.stop, "row_size")
        
        return f"""[&]() {{
    const auto& outer = {target};
    ROOT::RVec<ROOT::RVec<{elem_type}>> result;
    result.reserve(outer.size());
    for (size_t i = 0; i < outer.size(); ++i) {{
        const auto& row = outer[i];
        int row_size = static_cast<int>(row.size());
        int start = {start_expr};
        int stop = {stop_expr};
        // Normalize negative indices
        start = start >= 0 ? start : row_size + start;
        stop = stop >= 0 ? stop : row_size + stop;
        // Clamp to valid range
        start = std::max(0, std::min(start, row_size));
        stop = std::max(0, std::min(stop, row_size));
        ROOT::RVec<{elem_type}> slice;
        for (int k = start; k < stop; ++k) {{
            slice.push_back(row[k]);
        }}
        result.push_back(slice);
    }}
    return result;
}}()"""
    
    def _gen_column_slice_with_step(self, target: str, params: SliceParams,
                                     elem_type: str) -> str:
        """Generate column slice with step [::k] or [::-1]."""
        start_expr = self._format_slice_bound(params.start, "0")
        stop_expr = self._format_slice_bound(params.stop, "row_size")
        step_expr = self._format_index(params.step)
        
        # Handle reverse slice [::-1]
        if params.is_reverse():
            return f"""[&]() {{
    const auto& outer = {target};
    ROOT::RVec<ROOT::RVec<{elem_type}>> result;
    result.reserve(outer.size());
    for (size_t i = 0; i < outer.size(); ++i) {{
        const auto& row = outer[i];
        int row_size = static_cast<int>(row.size());
        ROOT::RVec<{elem_type}> slice;
        slice.reserve(row_size);
        for (int k = row_size - 1; k >= 0; --k) {{
            slice.push_back(row[k]);
        }}
        result.push_back(slice);
    }}
    return result;
}}()"""
        
        # General step slice
        return f"""[&]() {{
    const auto& outer = {target};
    ROOT::RVec<ROOT::RVec<{elem_type}>> result;
    result.reserve(outer.size());
    for (size_t i = 0; i < outer.size(); ++i) {{
        const auto& row = outer[i];
        int row_size = static_cast<int>(row.size());
        int start = {start_expr};
        int stop = {stop_expr};
        int step = {step_expr};
        // Normalize negative indices
        start = start >= 0 ? start : row_size + start;
        stop = stop >= 0 ? stop : row_size + stop;
        // Clamp to valid range
        start = std::max(0, std::min(start, row_size));
        stop = std::max(0, std::min(stop, row_size));
        if (step == 0) step = 1;  // Prevent infinite loop
        ROOT::RVec<{elem_type}> slice;
        if (step > 0) {{
            for (int k = start; k < stop; k += step) {{
                slice.push_back(row[k]);
            }}
        }} else {{
            for (int k = stop - 1; k >= start; k += step) {{
                slice.push_back(row[k]);
            }}
        }}
        result.push_back(slice);
    }}
    return result;
}}()"""
    
    # =========================================================================
    # Element and Row Access
    # =========================================================================
    
    def _gen_element_access(self, node: NestedAccessNode) -> str:
        """
        Generate code for element access: nested[i, j] → T.
        
        Uses fail-closed policy with NaN fallback for float types.
        """
        target = node.target
        row = self._format_index(node.row_index)
        col = self._format_index(node.col_index)
        elem_type = node.element_type
        
        # Use NaN for float types, throw for others
        if elem_type in ("double", "float"):
            fallback = f"std::numeric_limits<{elem_type}>::quiet_NaN()"
        else:
            fallback = f"throw std::runtime_error(\"Index out of bounds\")"
        
        self._dependencies.append("limits")
        
        return f"""[&]() {{
    const auto& outer = {target};
    int i = {row};
    int j = {col};
    int outer_size = static_cast<int>(outer.size());
    int i_norm = i >= 0 ? i : outer_size + i;
    if (i_norm < 0 || i_norm >= outer_size) {{
        return {fallback};
    }}
    const auto& inner = outer[i_norm];
    int inner_size = static_cast<int>(inner.size());
    int j_norm = j >= 0 ? j : inner_size + j;
    if (j_norm < 0 || j_norm >= inner_size) {{
        return {fallback};
    }}
    return static_cast<{elem_type}>(inner[j_norm]);
}}()"""
    
    def _gen_row_access(self, node: NestedAccessNode) -> str:
        """
        Generate code for row access: nested[i] → RVec<T>.
        
        Returns empty RVec for out-of-bounds row.
        """
        target = node.target
        row = self._format_index(node.row_index)
        elem_type = node.element_type
        
        return f"""[&]() {{
    const auto& outer = {target};
    int i = {row};
    int outer_size = static_cast<int>(outer.size());
    int i_norm = i >= 0 ? i : outer_size + i;
    if (i_norm < 0 || i_norm >= outer_size) {{
        return ROOT::RVec<{elem_type}>();
    }}
    return outer[i_norm];
}}()"""
    
    def _gen_row_slice(self, node: NestedAccessNode) -> str:
        """
        Generate code for row slicing: nested[:n] → RVec<RVec<T>>.
        """
        target = node.target
        elem_type = node.element_type
        
        row_slice = node.row_index if isinstance(node.row_index, SliceParams) else SliceParams()
        
        start_expr = self._format_slice_bound(row_slice.start, "0")
        stop_expr = self._format_slice_bound(row_slice.stop, "outer_size")
        
        return f"""[&]() {{
    const auto& outer = {target};
    int outer_size = static_cast<int>(outer.size());
    int start = {start_expr};
    int stop = {stop_expr};
    start = start >= 0 ? start : outer_size + start;
    stop = stop >= 0 ? stop : outer_size + stop;
    start = std::max(0, std::min(start, outer_size));
    stop = std::max(0, std::min(stop, outer_size));
    ROOT::RVec<ROOT::RVec<{elem_type}>> result;
    for (int i = start; i < stop; ++i) {{
        result.push_back(outer[i]);
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

def generate_nested_code(node: NestedAccessNode) -> GeneratedCode:
    """
    Generate C++ code for a nested access node.
    
    Convenience function that creates a generator and calls generate().
    
    Args:
        node: NestedAccessNode describing the operation
        
    Returns:
        GeneratedCode with code string, dependencies, and result type
    """
    generator = NestedCodeGenerator()
    return generator.generate(node)
