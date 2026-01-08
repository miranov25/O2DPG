"""
Phase 13.4.DSL D4-D7: Backend Code Generation for C-Arrays

Generates C++ code for C-array indexing and slicing operations.
Uses row-major stride calculation for multi-dimensional arrays.

FROZEN RULES:
- NO LAMBDA EXPRESSIONS - ROOT JIT crashes on them
- Use named JIT functions with macro guards
- All functions use const RVec<T>& parameters

Safety semantics (consistent with Phase 13.3):
- Scalar out-of-bounds: returns NaN
- Slice out-of-bounds: clamps to valid range
- Negative indices: normalized to positive
"""

from dataclasses import dataclass
from typing import Optional, List, Set
import hashlib

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
    jit_declarations: str = ""  # JIT function declarations to prepend


def _make_func_name(prefix: str, node: CArrayAccessNode) -> str:
    """Generate unique function name based on operation and parameters."""
    # Create a hash from the key parameters to ensure uniqueness
    key = f"{prefix}_{node.source}_{node.base_type}_{len(node.dims)}"
    for dim in node.dims:
        key += f"_{dim.size}_{dim.is_fixed}"
    hash_suffix = hashlib.md5(key.encode()).hexdigest()[:8]
    return f"carray_{prefix}_{hash_suffix}"


class CArrayCodeGenerator:
    """
    Generates C++ code for C-array access operations.
    
    FROZEN RULE: No lambdas! Uses named JIT functions with macro guards.
    Uses row-major stride: arr[i][j] → arr[i * cols + j]
    """
    
    def __init__(self):
        self._indent = "    "
    
    def generate(self, node: CArrayAccessNode) -> CArrayCodeResult:
        """Generate C++ code for the given IR node."""
        
        if node.kind == CArraySliceKind.ELEMENT_1D:
            code, jit_decl = self._gen_element_1d(node)
        elif node.kind == CArraySliceKind.SLICE_1D:
            code, jit_decl = self._gen_slice_1d(node)
        elif node.kind == CArraySliceKind.ELEMENT_2D:
            code, jit_decl = self._gen_element_2d(node)
        elif node.kind == CArraySliceKind.ROW_2D:
            code, jit_decl = self._gen_row_2d(node)
        elif node.kind == CArraySliceKind.COLUMN_2D:
            code, jit_decl = self._gen_column_2d(node)
        elif node.kind == CArraySliceKind.COLUMN_SLICE_2D:
            code, jit_decl = self._gen_column_slice_2d(node)
        elif node.kind == CArraySliceKind.ROW_SLICE_2D:
            code, jit_decl = self._gen_row_slice_2d(node)
        elif node.kind == CArraySliceKind.SUBARRAY_2D:
            code, jit_decl = self._gen_subarray_2d(node)
        elif node.kind == CArraySliceKind.ELEMENT_3D:
            code, jit_decl = self._gen_element_3d(node)
        elif node.kind == CArraySliceKind.PLANE_3D:
            code, jit_decl = self._gen_plane_3d(node)
        elif node.kind == CArraySliceKind.ROW_3D:
            code, jit_decl = self._gen_row_3d(node)
        elif node.kind == CArraySliceKind.SLICE_DIM0_3D:
            code, jit_decl = self._gen_slice_dim0_3d(node)
        elif node.kind == CArraySliceKind.SLICE_DIM1_3D:
            code, jit_decl = self._gen_slice_dim1_3d(node)
        elif node.kind == CArraySliceKind.PLANE_SLICE_3D:
            code, jit_decl = self._gen_plane_slice_3d(node)
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
            jit_declarations=jit_decl,
        )
    
    # =========================================================================
    # 1D Operations
    # =========================================================================
    
    def _gen_element_1d(self, node: CArrayAccessNode) -> tuple:
        """Generate 1D element access with bounds checking."""
        dim = node.dims[0]
        idx = node.indices[0]
        func_name = _make_func_name("elem1d", node)
        base_type = node.base_type
        
        size_expr = str(dim.size) if dim.is_fixed else dim.size
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
{base_type} {func_name}(const ROOT::RVec<{base_type}>& arr, int size, int i) {{
    int i_norm = i >= 0 ? i : size + i;
    if (i_norm < 0 || i_norm >= size) {{
        return std::numeric_limits<{base_type}>::quiet_NaN();
    }}
    return arr[i_norm];
}}
#endif
"""
        code = f"{func_name}({node.source}, {size_expr}, {idx})"
        return code, jit_decl
    
    def _gen_slice_1d(self, node: CArrayAccessNode) -> tuple:
        """Generate 1D slice with clamping."""
        dim = node.dims[0]
        sp = node.slices[0] if node.slices else SliceParams()
        func_name = _make_func_name("slice1d", node)
        base_type = node.base_type
        
        size_expr = str(dim.size) if dim.is_fixed else dim.size
        start_expr = str(sp.start) if sp.start is not None else "0"
        stop_expr = str(sp.stop) if sp.stop is not None else str(size_expr)
        step_expr = str(sp.step) if sp.step is not None else "1"
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<{base_type}> {func_name}(const ROOT::RVec<{base_type}>& arr, int size, int start, int stop, int step) {{
    // Normalize negative indices
    if (start < 0) start = size + start;
    if (stop < 0) stop = size + stop;
    
    // Clamp to valid range
    start = std::max(0, std::min(start, size));
    stop = std::max(0, std::min(stop, size));
    
    ROOT::RVec<{base_type}> result;
    if (step > 0) {{
        for (int k = start; k < stop; k += step) {{
            result.push_back(arr[k]);
        }}
    }} else {{
        for (int k = start; k > stop; k += step) {{
            result.push_back(arr[k]);
        }}
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {size_expr}, {start_expr}, {stop_expr}, {step_expr})"
        return code, jit_decl
    
    # =========================================================================
    # 2D Operations
    # =========================================================================
    
    def _gen_element_2d(self, node: CArrayAccessNode) -> tuple:
        """Generate 2D element access with row-major stride."""
        func_name = _make_func_name("elem2d", node)
        base_type = node.base_type
        
        rows_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        cols_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        row_idx = node.indices[0]
        col_idx = node.indices[1]
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
{base_type} {func_name}(const ROOT::RVec<{base_type}>& arr, int rows, int cols, int r, int c) {{
    int r_norm = r >= 0 ? r : rows + r;
    int c_norm = c >= 0 ? c : cols + c;
    if (r_norm < 0 || r_norm >= rows || c_norm < 0 || c_norm >= cols) {{
        return std::numeric_limits<{base_type}>::quiet_NaN();
    }}
    return arr[r_norm * cols + c_norm];
}}
#endif
"""
        code = f"{func_name}({node.source}, {rows_expr}, {cols_expr}, {row_idx}, {col_idx})"
        return code, jit_decl
    
    def _gen_row_2d(self, node: CArrayAccessNode) -> tuple:
        """Generate 2D row extraction."""
        func_name = _make_func_name("row2d", node)
        base_type = node.base_type
        
        rows_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        cols_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        row_idx = node.indices[0]
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<{base_type}> {func_name}(const ROOT::RVec<{base_type}>& arr, int rows, int cols, int r) {{
    int r_norm = r >= 0 ? r : rows + r;
    if (r_norm < 0 || r_norm >= rows) {{
        return ROOT::RVec<{base_type}>();
    }}
    ROOT::RVec<{base_type}> result;
    result.reserve(cols);
    int base = r_norm * cols;
    for (int c = 0; c < cols; ++c) {{
        result.push_back(arr[base + c]);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {rows_expr}, {cols_expr}, {row_idx})"
        return code, jit_decl
    
    def _gen_column_2d(self, node: CArrayAccessNode) -> tuple:
        """Generate 2D column extraction."""
        func_name = _make_func_name("col2d", node)
        base_type = node.base_type
        
        rows_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        cols_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        col_idx = node.indices[1]
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<{base_type}> {func_name}(const ROOT::RVec<{base_type}>& arr, int rows, int cols, int c) {{
    int c_norm = c >= 0 ? c : cols + c;
    if (c_norm < 0 || c_norm >= cols) {{
        return ROOT::RVec<{base_type}>();
    }}
    ROOT::RVec<{base_type}> result;
    result.reserve(rows);
    for (int r = 0; r < rows; ++r) {{
        result.push_back(arr[r * cols + c_norm]);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {rows_expr}, {cols_expr}, {col_idx})"
        return code, jit_decl
    
    def _gen_column_slice_2d(self, node: CArrayAccessNode) -> tuple:
        """Generate 2D column slice (multiple columns)."""
        func_name = _make_func_name("colslice2d", node)
        base_type = node.base_type
        
        rows_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        cols_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        col_slice = node.slices[1] if len(node.slices) > 1 and node.slices[1] else SliceParams()
        
        start_expr = str(col_slice.start) if col_slice.start is not None else "0"
        stop_expr = str(col_slice.stop) if col_slice.stop is not None else cols_expr
        step_expr = str(col_slice.step) if col_slice.step is not None else "1"
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<ROOT::RVec<{base_type}>> {func_name}(const ROOT::RVec<{base_type}>& arr, int rows, int cols, int c_start, int c_stop, int c_step) {{
    if (c_start < 0) c_start = cols + c_start;
    if (c_stop < 0) c_stop = cols + c_stop;
    c_start = std::max(0, std::min(c_start, cols));
    c_stop = std::max(0, std::min(c_stop, cols));
    
    ROOT::RVec<ROOT::RVec<{base_type}>> result;
    result.reserve(rows);
    for (int r = 0; r < rows; ++r) {{
        ROOT::RVec<{base_type}> row;
        if (c_step > 0) {{
            for (int c = c_start; c < c_stop; c += c_step) {{
                row.push_back(arr[r * cols + c]);
            }}
        }} else {{
            for (int c = c_start; c > c_stop; c += c_step) {{
                row.push_back(arr[r * cols + c]);
            }}
        }}
        result.push_back(row);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {rows_expr}, {cols_expr}, {start_expr}, {stop_expr}, {step_expr})"
        return code, jit_decl
    
    def _gen_row_slice_2d(self, node: CArrayAccessNode) -> tuple:
        """Generate 2D row slice (multiple rows)."""
        func_name = _make_func_name("rowslice2d", node)
        base_type = node.base_type
        
        rows_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        cols_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        row_slice = node.slices[0] if node.slices and node.slices[0] else SliceParams()
        
        start_expr = str(row_slice.start) if row_slice.start is not None else "0"
        stop_expr = str(row_slice.stop) if row_slice.stop is not None else rows_expr
        step_expr = str(row_slice.step) if row_slice.step is not None else "1"
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<ROOT::RVec<{base_type}>> {func_name}(const ROOT::RVec<{base_type}>& arr, int rows, int cols, int r_start, int r_stop, int r_step) {{
    if (r_start < 0) r_start = rows + r_start;
    if (r_stop < 0) r_stop = rows + r_stop;
    r_start = std::max(0, std::min(r_start, rows));
    r_stop = std::max(0, std::min(r_stop, rows));
    
    ROOT::RVec<ROOT::RVec<{base_type}>> result;
    if (r_step > 0) {{
        for (int r = r_start; r < r_stop; r += r_step) {{
            ROOT::RVec<{base_type}> row;
            row.reserve(cols);
            int base = r * cols;
            for (int c = 0; c < cols; ++c) {{
                row.push_back(arr[base + c]);
            }}
            result.push_back(row);
        }}
    }} else {{
        for (int r = r_start; r > r_stop; r += r_step) {{
            ROOT::RVec<{base_type}> row;
            row.reserve(cols);
            int base = r * cols;
            for (int c = 0; c < cols; ++c) {{
                row.push_back(arr[base + c]);
            }}
            result.push_back(row);
        }}
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {rows_expr}, {cols_expr}, {start_expr}, {stop_expr}, {step_expr})"
        return code, jit_decl
    
    def _gen_subarray_2d(self, node: CArrayAccessNode) -> tuple:
        """Generate 2D subarray (row and column slices)."""
        func_name = _make_func_name("subarray2d", node)
        base_type = node.base_type
        
        rows_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        cols_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        
        row_slice = node.slices[0] if node.slices and node.slices[0] else SliceParams()
        col_slice = node.slices[1] if len(node.slices) > 1 and node.slices[1] else SliceParams()
        
        r_start = str(row_slice.start) if row_slice.start is not None else "0"
        r_stop = str(row_slice.stop) if row_slice.stop is not None else rows_expr
        r_step = str(row_slice.step) if row_slice.step is not None else "1"
        
        c_start = str(col_slice.start) if col_slice.start is not None else "0"
        c_stop = str(col_slice.stop) if col_slice.stop is not None else cols_expr
        c_step = str(col_slice.step) if col_slice.step is not None else "1"
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<ROOT::RVec<{base_type}>> {func_name}(const ROOT::RVec<{base_type}>& arr, int rows, int cols, 
                                                 int r_start, int r_stop, int r_step,
                                                 int c_start, int c_stop, int c_step) {{
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
    
    ROOT::RVec<ROOT::RVec<{base_type}>> result;
    for (int r = r_start; r < r_stop; r += r_step) {{
        ROOT::RVec<{base_type}> row;
        for (int c = c_start; c < c_stop; c += c_step) {{
            row.push_back(arr[r * cols + c]);
        }}
        result.push_back(row);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {rows_expr}, {cols_expr}, {r_start}, {r_stop}, {r_step}, {c_start}, {c_stop}, {c_step})"
        return code, jit_decl
    
    # =========================================================================
    # 3D Operations
    # =========================================================================
    
    def _gen_element_3d(self, node: CArrayAccessNode) -> tuple:
        """Generate 3D element access with row-major stride."""
        func_name = _make_func_name("elem3d", node)
        base_type = node.base_type
        
        d0_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        d1_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        d2_expr = str(node.dims[2].size) if node.dims[2].is_fixed else node.dims[2].size
        i, j, k = node.indices[0], node.indices[1], node.indices[2]
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
{base_type} {func_name}(const ROOT::RVec<{base_type}>& arr, int d0, int d1, int d2, int i, int j, int k) {{
    int i_norm = i >= 0 ? i : d0 + i;
    int j_norm = j >= 0 ? j : d1 + j;
    int k_norm = k >= 0 ? k : d2 + k;
    if (i_norm < 0 || i_norm >= d0 || 
        j_norm < 0 || j_norm >= d1 || 
        k_norm < 0 || k_norm >= d2) {{
        return std::numeric_limits<{base_type}>::quiet_NaN();
    }}
    return arr[i_norm * d1 * d2 + j_norm * d2 + k_norm];
}}
#endif
"""
        code = f"{func_name}({node.source}, {d0_expr}, {d1_expr}, {d2_expr}, {i}, {j}, {k})"
        return code, jit_decl
    
    def _gen_plane_3d(self, node: CArrayAccessNode) -> tuple:
        """Generate 3D plane extraction (arr[i] → 2D slice)."""
        func_name = _make_func_name("plane3d", node)
        base_type = node.base_type
        
        d0_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        d1_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        d2_expr = str(node.dims[2].size) if node.dims[2].is_fixed else node.dims[2].size
        plane_idx = node.indices[0]
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<ROOT::RVec<{base_type}>> {func_name}(const ROOT::RVec<{base_type}>& arr, int d0, int d1, int d2, int i) {{
    int i_norm = i >= 0 ? i : d0 + i;
    if (i_norm < 0 || i_norm >= d0) {{
        return ROOT::RVec<ROOT::RVec<{base_type}>>();
    }}
    ROOT::RVec<ROOT::RVec<{base_type}>> result;
    result.reserve(d1);
    int plane_base = i_norm * d1 * d2;
    for (int j = 0; j < d1; ++j) {{
        ROOT::RVec<{base_type}> row;
        row.reserve(d2);
        int row_base = plane_base + j * d2;
        for (int k = 0; k < d2; ++k) {{
            row.push_back(arr[row_base + k]);
        }}
        result.push_back(row);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {d0_expr}, {d1_expr}, {d2_expr}, {plane_idx})"
        return code, jit_decl
    
    def _gen_row_3d(self, node: CArrayAccessNode) -> tuple:
        """Generate 3D row extraction (arr[i, j] → 1D slice)."""
        func_name = _make_func_name("row3d", node)
        base_type = node.base_type
        
        d0_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        d1_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        d2_expr = str(node.dims[2].size) if node.dims[2].is_fixed else node.dims[2].size
        i_idx = node.indices[0]
        j_idx = node.indices[1] if len(node.indices) > 1 else 0
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<{base_type}> {func_name}(const ROOT::RVec<{base_type}>& arr, int d0, int d1, int d2, int i, int j) {{
    int i_norm = i >= 0 ? i : d0 + i;
    int j_norm = j >= 0 ? j : d1 + j;
    if (i_norm < 0 || i_norm >= d0 || j_norm < 0 || j_norm >= d1) {{
        return ROOT::RVec<{base_type}>();
    }}
    ROOT::RVec<{base_type}> result;
    result.reserve(d2);
    int base = i_norm * d1 * d2 + j_norm * d2;
    for (int k = 0; k < d2; ++k) {{
        result.push_back(arr[base + k]);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {d0_expr}, {d1_expr}, {d2_expr}, {i_idx}, {j_idx})"
        return code, jit_decl
    
    def _gen_slice_dim0_3d(self, node: CArrayAccessNode) -> tuple:
        """Generate 3D slice along dim0 (arr[:, j, k] → 1D)."""
        func_name = _make_func_name("sliced0_3d", node)
        base_type = node.base_type
        
        d0_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        d1_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        d2_expr = str(node.dims[2].size) if node.dims[2].is_fixed else node.dims[2].size
        j_idx = node.indices[1]
        k_idx = node.indices[2]
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<{base_type}> {func_name}(const ROOT::RVec<{base_type}>& arr, int d0, int d1, int d2, int j, int k) {{
    int j_norm = j >= 0 ? j : d1 + j;
    int k_norm = k >= 0 ? k : d2 + k;
    if (j_norm < 0 || j_norm >= d1 || k_norm < 0 || k_norm >= d2) {{
        return ROOT::RVec<{base_type}>();
    }}
    ROOT::RVec<{base_type}> result;
    result.reserve(d0);
    for (int i = 0; i < d0; ++i) {{
        result.push_back(arr[i * d1 * d2 + j_norm * d2 + k_norm]);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {d0_expr}, {d1_expr}, {d2_expr}, {j_idx}, {k_idx})"
        return code, jit_decl
    
    def _gen_slice_dim1_3d(self, node: CArrayAccessNode) -> tuple:
        """Generate 3D slice along dim1 (arr[i, :, k] → 1D)."""
        func_name = _make_func_name("sliced1_3d", node)
        base_type = node.base_type
        
        d0_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        d1_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        d2_expr = str(node.dims[2].size) if node.dims[2].is_fixed else node.dims[2].size
        i_idx = node.indices[0]
        k_idx = node.indices[2]
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<{base_type}> {func_name}(const ROOT::RVec<{base_type}>& arr, int d0, int d1, int d2, int i, int k) {{
    int i_norm = i >= 0 ? i : d0 + i;
    int k_norm = k >= 0 ? k : d2 + k;
    if (i_norm < 0 || i_norm >= d0 || k_norm < 0 || k_norm >= d2) {{
        return ROOT::RVec<{base_type}>();
    }}
    ROOT::RVec<{base_type}> result;
    result.reserve(d1);
    int plane_base = i_norm * d1 * d2;
    for (int j = 0; j < d1; ++j) {{
        result.push_back(arr[plane_base + j * d2 + k_norm]);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {d0_expr}, {d1_expr}, {d2_expr}, {i_idx}, {k_idx})"
        return code, jit_decl
    
    def _gen_plane_slice_3d(self, node: CArrayAccessNode) -> tuple:
        """Generate 3D plane slice (arr[:, j, :] → 2D)."""
        func_name = _make_func_name("planeslice3d", node)
        base_type = node.base_type
        
        d0_expr = str(node.dims[0].size) if node.dims[0].is_fixed else node.dims[0].size
        d1_expr = str(node.dims[1].size) if node.dims[1].is_fixed else node.dims[1].size
        d2_expr = str(node.dims[2].size) if node.dims[2].is_fixed else node.dims[2].size
        j_idx = node.indices[1]
        
        jit_decl = f"""
#ifndef {func_name.upper()}_DEFINED
#define {func_name.upper()}_DEFINED
ROOT::RVec<ROOT::RVec<{base_type}>> {func_name}(const ROOT::RVec<{base_type}>& arr, int d0, int d1, int d2, int j) {{
    int j_norm = j >= 0 ? j : d1 + j;
    if (j_norm < 0 || j_norm >= d1) {{
        return ROOT::RVec<ROOT::RVec<{base_type}>>();
    }}
    ROOT::RVec<ROOT::RVec<{base_type}>> result;
    result.reserve(d0);
    for (int i = 0; i < d0; ++i) {{
        ROOT::RVec<{base_type}> row;
        row.reserve(d2);
        int base = i * d1 * d2 + j_norm * d2;
        for (int k = 0; k < d2; ++k) {{
            row.push_back(arr[base + k]);
        }}
        result.push_back(row);
    }}
    return result;
}}
#endif
"""
        code = f"{func_name}({node.source}, {d0_expr}, {d1_expr}, {d2_expr}, {j_idx})"
        return code, jit_decl


# =============================================================================
# Convenience Function
# =============================================================================

def generate_carray_code(node: CArrayAccessNode) -> CArrayCodeResult:
    """Generate C++ code for C-array access node."""
    generator = CArrayCodeGenerator()
    return generator.generate(node)
