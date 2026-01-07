# Diagnostics

This directory contains standalone validation scripts used during Phase 13.4 development.

These scripts are **not pytest tests** - they are exploratory/diagnostic tools that helped discover ROOT behavior and validate implementation strategies.

## Scripts

### `test_carray_dimension_diagnostic.py`

**Purpose**: Discover how ROOT handles multi-dimensional C-arrays in RDataFrame.

**Key Findings**:
- `float mat[3][4]` → `RVec<Float_t>` of size 12 (flattened, row-major)
- `float tensor[2][3][4]` → `RVec<Float_t>` of size 24 (flattened)
- Variable-length arrays `arr[n]/F` work correctly with counter branches

**Run**: `python diagnostics/test_carray_dimension_diagnostic.py`

### `test_nd_slicing_strategies.py`

**Purpose**: Validate all 7 ND slicing strategies work with ROOT's flattened arrays.

**Strategies Validated**:
1. Element access: `mat[i,j]` → `mat[i*cols+j]`
2. Row extraction: `mat[i,:]` (contiguous)
3. Column extraction: `mat[:,j]` (strided)
4. 3D plane: `tensor[i,:,:]`
5. 3D row: `tensor[i,j,:]`
6. 3D column: `tensor[i,:,k]`
7. Subarray: `mat[a:b,c:d]`

**Run**: `python diagnostics/test_nd_slicing_strategies.py`

## When to Use

- **Discovery**: Run when investigating ROOT/RDataFrame behavior
- **Validation**: Run after major changes to verify strategies still work
- **Documentation**: These scripts serve as executable documentation

## For CI/CD

The key validations from these scripts have been converted to pytest tests in:
- `tests/test_carray_root_integration.py` (16 tests, run with `-n 0`)
- `tests/test_carray_indexing.py` (51 tests, run with `-n 12`)

## Key Lessons Learned

1. **ROOT converts C-arrays to RVec** - No pointer handling needed
2. **Lambdas in Define() strings crash** - Use named functions instead
3. **pytest parallelization corrupts ROOT state** - Use `-n 0` for ROOT tests
4. **All helper functions use `const RVec<T>&`** - Not `T*`
