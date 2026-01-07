"""
Phase 13.4.DSL D4-D7: Backend Code Generation for C-Arrays

Generates C++ code for C-array indexing and slicing operations.
Uses row-major stride calculation for multi-dimensional arrays.

Safety semantics (consistent with Phase 13.3):
- Scalar out-of-bounds: returns NaN
- Slice out-of-bounds: clamps to valid range
- Negative indices: normalized to positive
"""

from dataclasses import dataclass
from typing import Optional, List, Set

from RDataFrameDSL.ir_nodes_carray import (
    CArrayAccessNode,
    CArraySliceKind,
    SliceParams,
)


__all__ = [
    'CArrayCodeResult',
    'CArrayCodeGenerator',
    'generate_carray_code',
]


@dataclass
class CArrayCodeResult:
    """Result of C-array code generation."""
    code: str
    result_type: str
    dependencies: Set[str]


class CArrayCodeGenerator:
    """
    Generates C++ code for C-array access operations.
    
    All generated code is lambda-wrapped for safe evaluation.
    Uses row-major stride: arr[i][j] → arr[i * cols + j]
    """
    
    def __init__(self):
        self._indent = "    "
    
    def generate(self, node: CArrayAccessNode) -> CArrayCodeResult:
        """Generate C++ code for the given IR node."""
        
        if node.kind == CArraySliceKind.ELEMENT_1D:
            code = self._gen_element_1d(node)
        elif node.kind == CArraySliceKind.SLICE_1D:
            code = self._gen_slice_1d(node)
        elif node.kind == CArraySliceKind.ELEMENT_2D:
            code = self._gen_element_2d(node)
        elif node.kind == CArraySliceKind.ROW_2D:
            code = self._gen_row_2d(node)
        elif node.kind == CArraySliceKind.COLUMN_2D:
            code = self._gen_column_2d(node)
        elif node.kind == CArraySliceKind.COLUMN_SLICE_2D:
            code = self._gen_column_slice_2d(node)
        elif node.kind == CArraySliceKind.ROW_SLICE_2D:
            code = self._gen_row_slice_2d(node)
        elif node.kind == CArraySliceKind.SUBARRAY_2D:
            code = self._gen_subarray_2d(node)
        elif node.kind == CArraySliceKind.ELEMENT_3D:
            code = self._gen_element_3d(node)
        elif node.kind == CArraySliceKind.PLANE_3D:
            code = self._gen_plane_3d(node)
        elif node.kind == CArraySliceKind.ROW_3D:
            code = self._gen_row_3d(node)
        elif node.kind == CArraySliceKind.SLICE_DIM0_3D:
            code = self._gen_slice_dim0_3d(node)
        elif node.kind == CArraySliceKind.SLICE_DIM1_3D:
            code = self._gen_slice_dim1_3d(node)
        elif node.kind == CArraySliceKind.PLANE_SLICE_3D:
            code = self._gen_plane_slice_3d(node)
        else:
            raise ValueError(f"Unknown CArraySliceKind: {node.kind}")
        
        deps = {node.source}
        # Add counter branches to dependencies
        for dim in node.dims:
            if not dim.is_fixed:
                deps.add(str(dim.size))
        
        return CArrayCodeResult(
            code=code,
            result_type=node.result_type,
            dependencies=deps,
        )
    
    # =========================================================================
    # 1D Operations
    # =========================================================================
    
    def _gen_element_1d(self, node: CArrayAccessNode) -> str:
        """Generate 1D element access with bounds checking."""
        dim = node.dims[0]
        idx = node.indices[0]
        
        size_decl = dim.get_decl("size")
        
        return f"""[&]() {{
    {size_decl};
    int i = {idx};
    int i_norm = i >= 0 ? i : size + i;
    if (i_norm < 0 || i_norm >= size) {{
        return std::numeric_limits<{node.base_type}>::quiet_NaN();
    }}
    return static_cast<{node.base_type}>({node.source}[i_norm]);
}}()"""
    
    def _gen_slice_1d(self, node: CArrayAccessNode) -> str:
        """Generate 1D slice with clamping."""
        dim = node.dims[0]
        sp = node.slices[0] if node.slices else SliceParams()
        
        size_decl = dim.get_decl("size")
        
        # Handle slice parameters
        start_expr = str(sp.start) if sp.start is not None else "0"
        stop_expr = str(sp.stop) if sp.stop is not None else "size"
        step_expr = str(sp.step) if sp.step is not None else "1"
        
        return f"""[&]() {{
    {size_decl};
    int start = {start_expr};
    int stop = {stop_expr};
    int step = {step_expr};
    
    // Normalize negative indices
    if (start < 0) start = size + start;
    if (stop < 0) stop = size + stop;
    
    // Clamp to valid range
    start = std::max(0, std::min(start, size));
    stop = std::max(0, std::min(stop, size));
    
    ROOT::RVec<{node.base_type}> result;
    if (step > 0) {{
        for (int k = start; k < stop; k += step) {{
            result.push_back({node.source}[k]);
        }}
    }} else {{
        // Reverse iteration
        for (int k = start; k > stop; k += step) {{
            result.push_back({node.source}[k]);
        }}
    }}
    return result;
}}()"""
    
    # =========================================================================
    # 2D Operations
    # =========================================================================
    
    def _gen_element_2d(self, node: CArrayAccessNode) -> str:
        """Generate 2D element access with row-major stride."""
        rows_decl = node.dims[0].get_decl("rows")
        cols_decl = node.dims[1].get_decl("cols")
        row_idx = node.indices[0]
        col_idx = node.indices[1]
        
        return f"""[&]() {{
    {rows_decl};
    {cols_decl};
    int r = {row_idx};
    int c = {col_idx};
    int r_norm = r >= 0 ? r : rows + r;
    int c_norm = c >= 0 ? c : cols + c;
    if (r_norm < 0 || r_norm >= rows || c_norm < 0 || c_norm >= cols) {{
        return std::numeric_limits<{node.base_type}>::quiet_NaN();
    }}
    return static_cast<{node.base_type}>({node.source}[r_norm * cols + c_norm]);
}}()"""
    
    def _gen_row_2d(self, node: CArrayAccessNode) -> str:
        """Generate 2D row extraction."""
        rows_decl = node.dims[0].get_decl("rows")
        cols_decl = node.dims[1].get_decl("cols")
        row_idx = node.indices[0]
        
        return f"""[&]() {{
    {rows_decl};
    {cols_decl};
    int r = {row_idx};
    int r_norm = r >= 0 ? r : rows + r;
    if (r_norm < 0 || r_norm >= rows) {{
        return ROOT::RVec<{node.base_type}>();
    }}
    ROOT::RVec<{node.base_type}> result;
    result.reserve(cols);
    int base = r_norm * cols;
    for (int c = 0; c < cols; ++c) {{
        result.push_back({node.source}[base + c]);
    }}
    return result;
}}()"""
    
    def _gen_column_2d(self, node: CArrayAccessNode) -> str:
        """Generate 2D column extraction."""
        rows_decl = node.dims[0].get_decl("rows")
        cols_decl = node.dims[1].get_decl("cols")
        col_idx = node.indices[1]
        
        return f"""[&]() {{
    {rows_decl};
    {cols_decl};
    int c = {col_idx};
    int c_norm = c >= 0 ? c : cols + c;
    if (c_norm < 0 || c_norm >= cols) {{
        return ROOT::RVec<{node.base_type}>();
    }}
    ROOT::RVec<{node.base_type}> result;
    result.reserve(rows);
    for (int r = 0; r < rows; ++r) {{
        result.push_back({node.source}[r * cols + c_norm]);
    }}
    return result;
}}()"""
    
    def _gen_column_slice_2d(self, node: CArrayAccessNode) -> str:
        """Generate 2D column slice (multiple columns)."""
        rows_decl = node.dims[0].get_decl("rows")
        cols_decl = node.dims[1].get_decl("cols")
        col_slice = node.slices[1] if len(node.slices) > 1 and node.slices[1] else SliceParams()
        
        start_expr = str(col_slice.start) if col_slice.start is not None else "0"
        stop_expr = str(col_slice.stop) if col_slice.stop is not None else "cols"
        step_expr = str(col_slice.step) if col_slice.step is not None else "1"
        
        return f"""[&]() {{
    {rows_decl};
    {cols_decl};
    int c_start = {start_expr};
    int c_stop = {stop_expr};
    int c_step = {step_expr};
    
    if (c_start < 0) c_start = cols + c_start;
    if (c_stop < 0) c_stop = cols + c_stop;
    c_start = std::max(0, std::min(c_start, cols));
    c_stop = std::max(0, std::min(c_stop, cols));
    
    ROOT::RVec<ROOT::RVec<{node.base_type}>> result;
    result.reserve(rows);
    for (int r = 0; r < rows; ++r) {{
        ROOT::RVec<{node.base_type}> row;
        if (c_step > 0) {{
            for (int c = c_start; c < c_stop; c += c_step) {{
                row.push_back({node.source}[r * cols + c]);
            }}
        }} else {{
            for (int c = c_start; c > c_stop; c += c_step) {{
                row.push_back({node.source}[r * cols + c]);
            }}
        }}
        result.push_back(row);
    }}
    return result;
}}()"""
    
    def _gen_row_slice_2d(self, node: CArrayAccessNode) -> str:
        """Generate 2D row slice (multiple rows)."""
        rows_decl = node.dims[0].get_decl("rows")
        cols_decl = node.dims[1].get_decl("cols")
        row_slice = node.slices[0] if node.slices and node.slices[0] else SliceParams()
        
        start_expr = str(row_slice.start) if row_slice.start is not None else "0"
        stop_expr = str(row_slice.stop) if row_slice.stop is not None else "rows"
        step_expr = str(row_slice.step) if row_slice.step is not None else "1"
        
        return f"""[&]() {{
    {rows_decl};
    {cols_decl};
    int r_start = {start_expr};
    int r_stop = {stop_expr};
    int r_step = {step_expr};
    
    if (r_start < 0) r_start = rows + r_start;
    if (r_stop < 0) r_stop = rows + r_stop;
    r_start = std::max(0, std::min(r_start, rows));
    r_stop = std::max(0, std::min(r_stop, rows));
    
    ROOT::RVec<ROOT::RVec<{node.base_type}>> result;
    if (r_step > 0) {{
        for (int r = r_start; r < r_stop; r += r_step) {{
            ROOT::RVec<{node.base_type}> row;
            row.reserve(cols);
            int base = r * cols;
            for (int c = 0; c < cols; ++c) {{
                row.push_back({node.source}[base + c]);
            }}
            result.push_back(row);
        }}
    }} else {{
        for (int r = r_start; r > r_stop; r += r_step) {{
            ROOT::RVec<{node.base_type}> row;
            row.reserve(cols);
            int base = r * cols;
            for (int c = 0; c < cols; ++c) {{
                row.push_back({node.source}[base + c]);
            }}
            result.push_back(row);
        }}
    }}
    return result;
}}()"""
    
    def _gen_subarray_2d(self, node: CArrayAccessNode) -> str:
        """Generate 2D subarray (row and column slices)."""
        rows_decl = node.dims[0].get_decl("rows")
        cols_decl = node.dims[1].get_decl("cols")
        
        row_slice = node.slices[0] if node.slices and node.slices[0] else SliceParams()
        col_slice = node.slices[1] if len(node.slices) > 1 and node.slices[1] else SliceParams()
        
        r_start = str(row_slice.start) if row_slice.start is not None else "0"
        r_stop = str(row_slice.stop) if row_slice.stop is not None else "rows"
        r_step = str(row_slice.step) if row_slice.step is not None else "1"
        
        c_start = str(col_slice.start) if col_slice.start is not None else "0"
        c_stop = str(col_slice.stop) if col_slice.stop is not None else "cols"
        c_step = str(col_slice.step) if col_slice.step is not None else "1"
        
        return f"""[&]() {{
    {rows_decl};
    {cols_decl};
    int r_start = {r_start}, r_stop = {r_stop}, r_step = {r_step};
    int c_start = {c_start}, c_stop = {c_stop}, c_step = {c_step};
    
    // Normalize and clamp rows
    if (r_start < 0) r_start = rows + r_start;
    if (r_stop < 0) r_stop = rows + r_stop;
    r_start = std::max(0, std::min(r_start, rows));
    r_stop = std::max(0, std::min(r_stop, rows));
    
    // Normalize and clamp cols
    if (c_start < 0) c_start = cols + c_start;
    if (c_stop < 0) c_stop = cols + c_stop;
    c_start = std::max(0, std::min(c_start, cols));
    c_stop = std::max(0, std::min(c_stop, cols));
    
    ROOT::RVec<ROOT::RVec<{node.base_type}>> result;
    for (int r = r_start; r < r_stop; r += r_step) {{
        ROOT::RVec<{node.base_type}> row;
        for (int c = c_start; c < c_stop; c += c_step) {{
            row.push_back({node.source}[r * cols + c]);
        }}
        result.push_back(row);
    }}
    return result;
}}()"""
    
    # =========================================================================
    # 3D Operations
    # =========================================================================
    
    def _gen_element_3d(self, node: CArrayAccessNode) -> str:
        """Generate 3D element access with row-major stride."""
        d0_decl = node.dims[0].get_decl("d0")
        d1_decl = node.dims[1].get_decl("d1")
        d2_decl = node.dims[2].get_decl("d2")
        i, j, k = node.indices[0], node.indices[1], node.indices[2]
        
        return f"""[&]() {{
    {d0_decl};
    {d1_decl};
    {d2_decl};
    int i = {i}, j = {j}, k = {k};
    int i_norm = i >= 0 ? i : d0 + i;
    int j_norm = j >= 0 ? j : d1 + j;
    int k_norm = k >= 0 ? k : d2 + k;
    if (i_norm < 0 || i_norm >= d0 || 
        j_norm < 0 || j_norm >= d1 || 
        k_norm < 0 || k_norm >= d2) {{
        return std::numeric_limits<{node.base_type}>::quiet_NaN();
    }}
    return static_cast<{node.base_type}>({node.source}[i_norm * d1 * d2 + j_norm * d2 + k_norm]);
}}()"""
    
    def _gen_plane_3d(self, node: CArrayAccessNode) -> str:
        """Generate 3D plane extraction (arr[i] → 2D slice)."""
        d0_decl = node.dims[0].get_decl("d0")
        d1_decl = node.dims[1].get_decl("d1")
        d2_decl = node.dims[2].get_decl("d2")
        plane_idx = node.indices[0]
        
        return f"""[&]() {{
    {d0_decl};
    {d1_decl};
    {d2_decl};
    int i = {plane_idx};
    int i_norm = i >= 0 ? i : d0 + i;
    if (i_norm < 0 || i_norm >= d0) {{
        return ROOT::RVec<ROOT::RVec<{node.base_type}>>();
    }}
    ROOT::RVec<ROOT::RVec<{node.base_type}>> result;
    result.reserve(d1);
    int plane_base = i_norm * d1 * d2;
    for (int j = 0; j < d1; ++j) {{
        ROOT::RVec<{node.base_type}> row;
        row.reserve(d2);
        int row_base = plane_base + j * d2;
        for (int k = 0; k < d2; ++k) {{
            row.push_back({node.source}[row_base + k]);
        }}
        result.push_back(row);
    }}
    return result;
}}()"""
    
    def _gen_row_3d(self, node: CArrayAccessNode) -> str:
        """Generate 3D row extraction (arr[i, j] → 1D slice)."""
        d0_decl = node.dims[0].get_decl("d0")
        d1_decl = node.dims[1].get_decl("d1")
        d2_decl = node.dims[2].get_decl("d2")
        i_idx = node.indices[0]
        j_idx = node.indices[1] if len(node.indices) > 1 else 0
        
        return f"""[&]() {{
    {d0_decl};
    {d1_decl};
    {d2_decl};
    int i = {i_idx}, j = {j_idx};
    int i_norm = i >= 0 ? i : d0 + i;
    int j_norm = j >= 0 ? j : d1 + j;
    if (i_norm < 0 || i_norm >= d0 || j_norm < 0 || j_norm >= d1) {{
        return ROOT::RVec<{node.base_type}>();
    }}
    ROOT::RVec<{node.base_type}> result;
    result.reserve(d2);
    int base = i_norm * d1 * d2 + j_norm * d2;
    for (int k = 0; k < d2; ++k) {{
        result.push_back({node.source}[base + k]);
    }}
    return result;
}}()"""
    
    def _gen_slice_dim0_3d(self, node: CArrayAccessNode) -> str:
        """Generate 3D slice along dim0 (arr[:, j, k] → 1D)."""
        d0_decl = node.dims[0].get_decl("d0")
        d1_decl = node.dims[1].get_decl("d1")
        d2_decl = node.dims[2].get_decl("d2")
        j_idx = node.indices[1]
        k_idx = node.indices[2]
        
        return f"""[&]() {{
    {d0_decl};
    {d1_decl};
    {d2_decl};
    int j = {j_idx}, k = {k_idx};
    int j_norm = j >= 0 ? j : d1 + j;
    int k_norm = k >= 0 ? k : d2 + k;
    if (j_norm < 0 || j_norm >= d1 || k_norm < 0 || k_norm >= d2) {{
        return ROOT::RVec<{node.base_type}>();
    }}
    ROOT::RVec<{node.base_type}> result;
    result.reserve(d0);
    for (int i = 0; i < d0; ++i) {{
        result.push_back({node.source}[i * d1 * d2 + j_norm * d2 + k_norm]);
    }}
    return result;
}}()"""
    
    def _gen_slice_dim1_3d(self, node: CArrayAccessNode) -> str:
        """Generate 3D slice along dim1 (arr[i, :, k] → 1D)."""
        d0_decl = node.dims[0].get_decl("d0")
        d1_decl = node.dims[1].get_decl("d1")
        d2_decl = node.dims[2].get_decl("d2")
        i_idx = node.indices[0]
        k_idx = node.indices[2]
        
        return f"""[&]() {{
    {d0_decl};
    {d1_decl};
    {d2_decl};
    int i = {i_idx}, k = {k_idx};
    int i_norm = i >= 0 ? i : d0 + i;
    int k_norm = k >= 0 ? k : d2 + k;
    if (i_norm < 0 || i_norm >= d0 || k_norm < 0 || k_norm >= d2) {{
        return ROOT::RVec<{node.base_type}>();
    }}
    ROOT::RVec<{node.base_type}> result;
    result.reserve(d1);
    int plane_base = i_norm * d1 * d2;
    for (int j = 0; j < d1; ++j) {{
        result.push_back({node.source}[plane_base + j * d2 + k_norm]);
    }}
    return result;
}}()"""
    
    def _gen_plane_slice_3d(self, node: CArrayAccessNode) -> str:
        """Generate 3D plane slice (arr[:, j, :] → 2D)."""
        d0_decl = node.dims[0].get_decl("d0")
        d1_decl = node.dims[1].get_decl("d1")
        d2_decl = node.dims[2].get_decl("d2")
        j_idx = node.indices[1]
        
        return f"""[&]() {{
    {d0_decl};
    {d1_decl};
    {d2_decl};
    int j = {j_idx};
    int j_norm = j >= 0 ? j : d1 + j;
    if (j_norm < 0 || j_norm >= d1) {{
        return ROOT::RVec<ROOT::RVec<{node.base_type}>>();
    }}
    ROOT::RVec<ROOT::RVec<{node.base_type}>> result;
    result.reserve(d0);
    for (int i = 0; i < d0; ++i) {{
        ROOT::RVec<{node.base_type}> row;
        row.reserve(d2);
        int base = i * d1 * d2 + j_norm * d2;
        for (int k = 0; k < d2; ++k) {{
            row.push_back({node.source}[base + k]);
        }}
        result.push_back(row);
    }}
    return result;
}}()"""


# =============================================================================
# Convenience Function
# =============================================================================

def generate_carray_code(node: CArrayAccessNode) -> CArrayCodeResult:
    """Generate C++ code for C-array access node."""
    generator = CArrayCodeGenerator()
    return generator.generate(node)
