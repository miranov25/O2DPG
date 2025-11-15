# AliasDataFrame Compression Guide

## Overview

AliasDataFrame supports bidirectional column compression to reduce memory usage and file size while maintaining data accessibility through lazy decompression.

**Key Benefits:**
- 35-40% file size reduction
- Reversible compression (no data structure loss)
- Sub-micrometer precision for spatial coordinates
- Lazy decompression via aliases
- Flexible missing column handling

---

## Quick Start

### Basic Compression

```python
from dfextensions.AliasDataFrame import AliasDataFrame
import numpy as np

# Define compression schema
spec = {
    'dy': {
        'compress': 'round(asinh(dy)*40)',      # Transform for compression
        'decompress': 'sinh(dy_c/40.)',         # Transform for decompression
        'compressed_dtype': np.int16,           # Storage dtype
        'decompressed_dtype': np.float16        # Reconstructed dtype
    }
}

# Compress column
adf = AliasDataFrame(df)
adf.compress_columns(spec)  # Warns and skips missing columns by default

# Access decompressed values (via alias)
dy_values = adf.dy  # Automatically decompressed

# Save (aliases become ROOT TTree aliases)
adf.export_tree("output.root", "tree")
```

---

## NEW: Flexible Missing Column Handling

Starting with v1.2.0, `compress_columns` provides flexible handling of missing columns through the `on_missing` parameter.

### Default Behavior (on_missing='warn')

```python
# Spec includes columns that may not exist yet
spec = {
    'dy': {...},
    'dz': {...},
    'calibration_factor': {...}  # May not exist in all datasets
}

# Default: warns and continues with available columns
adf.compress_columns(spec)
# Warning: Skipping missing columns: ['calibration_factor']
# Result: dy and dz compressed, calibration_factor skipped
```

### Strict Mode (on_missing='error')

```python
# Production workflows: fail if columns missing
adf.compress_columns(spec, on_missing='error')
# Raises: KeyError("Missing columns: ['calibration_factor']")
```

### Silent Mode (on_missing='ignore')

```python
# Automated scripts: no warnings
adf.compress_columns(spec, on_missing='ignore')
# Silently skips missing columns, no output
```

### Return Summary

```python
# Get detailed results
result = adf.compress_columns(spec, return_summary=True)
print(f"Compressed: {result['compressed']}")  # ['dy', 'dz']
print(f"Skipped: {result['skipped']}")        # ['calibration_factor']
```

---

## Compression Modes

### Mode 1: Define Schema First (Pattern 1)
```python
# Step 1: Define schema upfront
adf.define_compression_schema(spec)

# Step 2: Compress when data ready
adf.compress_columns(columns=['dy', 'dz'])
```

**Use Case:** Known schema, compress incrementally as data arrives

---

### Mode 2: On-Demand Compression (Pattern 2)
```python
# Compress only specific columns
adf.compress_columns(spec, columns=['dy', 'dz'])  # Only dy, dz

# Later, add more columns
adf.compress_columns(spec, columns=['tgSlp'])     # Add tgSlp
```

**Use Case:** Incremental development, selective compression

---

### Mode 3: Compress All Available
```python
# Compress all columns in spec that exist in DataFrame
adf.compress_columns(spec)  # Default: warns on missing, compresses available
```

**Use Case:** Compress entire dataset, skip missing columns

---

### Mode 4: Strict Compression (NEW)
```python
# Fail if any column missing (production workflows)
adf.compress_columns(spec, on_missing='error')
```

**Use Case:** Validation, ensuring data quality

---

## Progressive Workflow Example

```python
# Day 1: Initial data collection (only dy, dz available)
spec = {
    'dy': {...},
    'dz': {...},
    'calibration': {...},  # Not available yet
    'temperature': {...}   # Not available yet
}

result = adf.compress_columns(spec, return_summary=True)
# Compressed: ['dy', 'dz']
# Skipped: ['calibration', 'temperature']

# Day 2: Calibration data arrives
adf_with_calib = AliasDataFrame(df_with_calibration)
result = adf_with_calib.compress_columns(spec, return_summary=True)
# Compressed: ['dy', 'dz', 'calibration']
# Skipped: ['temperature']

# Day 3: Temperature data arrives - compress all
adf_complete = AliasDataFrame(df_complete)
result = adf_complete.compress_columns(spec, return_summary=True)
# Compressed: ['dy', 'dz', 'calibration', 'temperature']
# Skipped: []
```

---

## State Management

### Compression States

Each column has one of these states:
- **COMPRESSED** - Column stored compressed, accessible via alias
- **DECOMPRESSED** - Column materialized, schema retained
- **SCHEMA_ONLY** - Schema defined, not yet compressed

### State Transitions

```
None ──────────────────► COMPRESSED
  │                      │
  └──► SCHEMA_ONLY ──────┤
                         │
                         ▼
                   DECOMPRESSED
                         │
                         └──────► COMPRESSED (recompression)
```

### Checking State

```python
# Check if column is compressed
if adf.is_compressed('dy'):
    print("dy is compressed")

# Get detailed state
state = adf.get_compression_state('dy')  # Returns 'compressed', 'decompressed', 'schema_only', or None

# View all compression info
info = adf.get_compression_info()
print(info)
```

---

## Decompression

### Basic Decompression

```python
# Decompress columns (keeps schema for recompression)
adf.decompress_columns(['dy', 'dz'])

# Remove schema entirely
adf.decompress_columns(['dy'], keep_schema=False, keep_compressed=False)
```

### Recompression

```python
# After decompression, can recompress with stored schema
adf.decompress_columns(['dy'])
# ... modify data ...
adf.compress_columns(columns=['dy'])  # Uses stored schema
```

---

## Precision Measurement

```python
# Measure compression quality
adf.compress_columns(spec, measure_precision=True)

# View precision info
info = adf.get_compression_info()
print(f"RMSE: {info['dy']['precision']['rmse']}")
print(f"Max error: {info['dy']['precision']['max_error']}")
```

**Metrics provided:**
- RMSE (root mean squared error)
- Max absolute error
- Mean error
- Sample counts (total vs finite)

---

## Common Patterns

### Pattern: Incremental Data Collection with Flexible Handling

```python
# Define schema for all expected columns
full_spec = {
    'dy': {...}, 'dz': {...},
    'y': {...}, 'z': {...},
    'tgSlp': {...}, 'mP3': {...}
}

# Day 1: Compress only available columns (default behavior)
result = adf.compress_columns(full_spec, return_summary=True)
print(f"Compressed {len(result['compressed'])} columns")
print(f"Waiting for {len(result['skipped'])} columns")

# Day 2: Re-run same code - newly arrived columns compressed
adf2 = AliasDataFrame(updated_df)
result = adf2.compress_columns(full_spec, return_summary=True)
# Automatically handles new columns
```

### Pattern: Validated Production Workflow

```python
# Production: ensure all required columns present
required_columns = ['dy', 'dz', 'y', 'z']
spec_required = {k: full_spec[k] for k in required_columns}

try:
    adf.compress_columns(spec_required, on_missing='error')
    print("✓ All required columns compressed")
except KeyError as e:
    print(f"✗ Missing required columns: {e}")
    # Fail fast - data quality issue
```

### Pattern: Schema Refinement

```python
# V1: Initial compression
adf.compress_columns(v1_spec, columns=['dy'])

# Decompress to refine
adf.decompress_columns(['dy'], keep_schema=False)

# V2: Improved compression
adf.compress_columns(v2_spec, columns=['dy'])
```

### Pattern: Debugging with Summary

```python
# Development: track what's happening
result = adf.compress_columns(spec, return_summary=True, on_missing='ignore')

print(f"Successfully compressed: {result['compressed']}")
print(f"Skipped (missing): {result['skipped']}")

# Verify expectations
expected = ['dy', 'dz', 'y']
if set(result['compressed']) != set(expected):
    print(f"Warning: Expected {expected}, got {result['compressed']}")
```

---

## Best Practices

### ✅ DO

1. **Use default 'warn' mode for development** - Flexible, informative
2. **Use 'error' mode for production** - Fail fast on missing data
3. **Use 'ignore' mode for automation** - Clean logs
4. **Use return_summary for debugging** - Track compression results
5. **Define schema once** - Centralize compression definitions
6. **Measure precision** - Verify acceptable error for your use case
7. **Use asinh for residuals** - Handles outliers well

### ❌ DON'T

1. **Don't ignore warnings in production** - They indicate data issues
2. **Don't use 'ignore' mode without logging** - Silently skipping = debugging nightmare
3. **Don't compress categorical data** - Use original values
4. **Don't change dtype mid-workflow** - Stick to schema
5. **Don't compress derived columns** - Keep computation in aliases
6. **Don't nest compression** - One level only

---

## Real-World Example: TPC Residuals

```python
# Define compression schema (once, centrally)
dfResCompresion = {
    'dy': {
        'compress': 'round(asinh(dy)*40)',
        'decompress': 'sinh(dy_c/40.)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float16
    },
    'dz': {
        'compress': 'round(asinh(dz)*40)',
        'decompress': 'sinh(dz_c/40.)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float16
    },
    'y': {
        'compress': 'round(y*(0x7fff/50))',
        'decompress': 'y_c*(50/0x7fff)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float32
    },
    'z': {
        'compress': 'round(z*(0x7fff/300))',
        'decompress': 'z_c*(300/0x7fff)',
        'compressed_dtype': np.int16,
        'decompressed_dtype': np.float32
    },
    # ... more columns
}

# Compress dataset (warns on missing, compresses available)
adf = AliasDataFrame(df_residuals)
result = adf.compress_columns(dfResCompresion, 
                               measure_precision=True,
                               return_summary=True)

print(f"Compressed: {result['compressed']}")
print(f"Skipped: {result['skipped']}")

# Export (508 MB → 330 MB, 35% reduction)
adf.export_tree("residuals_compressed.root", "tree")

# Later: Load and use (aliases decompress automatically)
adf_loaded = AliasDataFrame.import_tree("residuals_compressed.root", "tree")
dy_values = adf_loaded.dy  # Decompressed on-the-fly
```

**Results:**
- File size: 508 MB → 330 MB (35% reduction)
- Memory: 1579 MB → 1471 MB (7% reduction)
- Precision: RMSE < 0.018 mm for residuals
- Processing: <30 seconds for 9.6M rows

---

## Troubleshooting

### Warning: "Skipping missing columns"

```python
# This is normal in progressive workflows
result = adf.compress_columns(spec, return_summary=True)
print(f"Missing: {result['skipped']}")

# If unexpected, use error mode to debug:
adf.compress_columns(spec, on_missing='error')  # Will raise KeyError
```

### Error: "Missing columns: [...]"

```python
# Problem: Using error mode with incomplete data
# Solution 1: Use default warn mode
adf.compress_columns(spec)  # Warns, continues

# Solution 2: Compress only subset
adf.compress_columns(spec, columns=['dy', 'dz'])  # Only existing cols

# Solution 3: Define schema, compress later
adf.define_compression_schema(spec)  # Schema only
# ... later when data exists ...
adf.compress_columns(columns=['dy'])
```

### Error: "Column already compressed"

```python
# Problem: Trying to compress COMPRESSED column with different schema
# Solution: Decompress first or use selective mode (idempotent if same schema)
adf.decompress_columns(['dy'])
adf.compress_columns(spec, columns=['dy'])
```

---

## API Reference

### Compression Methods

```python
# Compress columns
adf.compress_columns(compression_spec=None, columns=None, 
                     suffix='_c', drop_original=True,
                     on_missing='warn',          # NEW in v1.2.0
                     return_summary=False,       # NEW in v1.2.0
                     measure_precision=False)

# Decompress columns
adf.decompress_columns(columns=None, keep_compressed=True, 
                       keep_schema=True)

# Define schema without compressing
adf.define_compression_schema(compression_spec, suffix='_c')
```

### New Parameters (v1.2.0)

**on_missing**: `{'warn', 'error', 'ignore'}`, default='warn'
- How to handle columns that don't exist in DataFrame
- 'warn': Skip with warning (recommended for development)
- 'error': Raise KeyError (recommended for production)
- 'ignore': Skip silently (use for automation)

**return_summary**: `bool`, default=False
- If True, return dict with 'compressed' and 'skipped' column lists
- If False, return self (backward compatible, method chaining)

### Introspection Methods

```python
# Check if compressed
is_compressed = adf.is_compressed('column_name')

# Get state
state = adf.get_compression_state('column_name')

# Get all compression info
info = adf.get_compression_info()  # Returns DataFrame

# Get single column info
info = adf.get_compression_info('column_name')  # Returns dict
```

---

## Version History

### v1.2.0 (Current)
- **NEW:** Flexible missing column handling (`on_missing` parameter)
- **NEW:** Compression summary (`return_summary` parameter)
- **BREAKING CHANGE:** Default behavior now warns on missing columns (was: error)
  - Migration: Use `on_missing='error'` for old strict behavior
- Smart column availability checking (df.columns + aliases + compression_info)
- Uniform filtering across all modes

### v1.1.0
- Selective compression mode (Pattern 2)
- Idempotent compression
- Schema updates
- Enhanced validation

### v1.0.0
- Basic compression/decompression
- State machine with 3 states
- Precision measurement
- Schema persistence

---

## Migration Guide (v1.1.0 → v1.2.0)

### Breaking Change: Default Error Behavior

**Old behavior (v1.1.0):**
```python
# Raised error on missing columns
adf.compress_columns(spec)  # KeyError if column missing
```

**New behavior (v1.2.0):**
```python
# Warns and continues by default
adf.compress_columns(spec)  # Warning, skips missing

# Use error mode for old behavior:
adf.compress_columns(spec, on_missing='error')  # Same as v1.1.0
```

### Recommended Migration

```python
# Development/Incremental workflows - use default
adf.compress_columns(spec)  # Flexible, warns on missing

# Production/Validation - use error mode
adf.compress_columns(spec, on_missing='error')  # Strict validation

# Automation - use ignore mode + summary
result = adf.compress_columns(spec, on_missing='ignore', return_summary=True)
if result['skipped']:
    logging.info(f"Skipped columns: {result['skipped']}")
```

---

## See Also

- **USER_GUIDE.md** - Complete feature overview
- **CHANGELOG.md** - Detailed version history
- **API_REFERENCE.md** - Complete API documentation
