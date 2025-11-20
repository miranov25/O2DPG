# Changelog

All notable changes to AliasDataFrame will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- **Non-materializing alias evaluation** - New methods to evaluate aliases without storing as DataFrame columns
  - `get_alias_series(name, dtype)` → Returns pandas Series without adding column
  - `get_alias_array(name, dtype)` → Returns numpy array without adding column
  - Enables clean boolean selections and temporary computations
  - Dependencies auto-materialize (consistent with existing behavior)
  - Robust handling of scalars (broadcast), arrays (validate length), Series
  - Comprehensive error handling (KeyError, ValueError, TypeError)
  - Support for subframe aliases (foo.bar notation)
  - Does not change existing alias semantics: only dependencies may be materialized; evaluated alias is never added as a column

### Use Cases
- Boolean selection masks without DataFrame pollution
- Temporary computations for analysis/visualization  
- Unified treatment of selections as boolean aliases (no separate structure needed)
- Clean DataFrames in progressive calibration workflows

### Examples
```python
# Define selection as boolean alias
adf.add_alias("highPt", "pt > 5.0", dtype=bool)

# Get mask without adding column
mask = adf.get_alias_array("highPt")
selected = adf.df[mask]

# DataFrame stays clean
assert 'highPt' not in adf.df.columns  # ✓
```

### Testing
- All existing tests passing (70+)
- Validated with TPC calibration workflow
- Reviewed and approved by GPT

---

## [1.1.0] - 2025-01-09

### Added
- **Selective compression mode (Pattern 2)** - Compress specific columns from a larger schema
  - New API: `compress_columns(spec, columns=['dy', 'dz'])`
  - Enables incremental compression workflows
  - Only specified columns are registered and compressed
- **Idempotent compression** - Re-compressing with same schema is safe (no-op)
  - Prevents errors in automation and scripting
  - Useful for incremental data collection
- **Schema updates** - Update compression schema for specific columns
  - Works for SCHEMA_ONLY and DECOMPRESSED states
  - Errors on COMPRESSED state (must decompress first)
- **Enhanced validation** - Column existence checked before compression
  - Clear error messages with available columns listed
  - Validates columns present in compression spec
- **Pattern mixing support** - Combine Pattern 1 and Pattern 2
  - Pattern 1: Schema-first (define all, compress incrementally)
  - Pattern 2: On-demand (compress as needed)
  - Column-local schema semantics (schemas can diverge)

### Changed
- `compress_columns()` now supports 5 modes (previously 3):
  1. Schema-only definition: `compress_columns(spec, columns=[])`
  2. Apply existing schema: `compress_columns(columns=['dy'])`
  3. Compress all in spec: `compress_columns(spec)`
  4. **Selective compression (NEW)**: `compress_columns(spec, columns=['dy', 'dz'])`
  5. Auto-compress eligible: `compress_columns()`
- Improved error messages for compression failures
  - Specific guidance for state transition errors
  - Clear suggestions for resolution
- Updated documentation with comprehensive examples

### Fixed
- None (fully backward compatible)

### Performance
- Negligible overhead from new validation (~O(1) dict lookups)
- No regression in existing compression performance
- Validated with 9.6M row TPC residual dataset

### Documentation
- Added `docs/COMPRESSION_GUIDE.md` with comprehensive usage guide
- Updated method docstrings with Pattern 2 examples
- Added state machine documentation
- Added troubleshooting section

### Testing
- Added 10 comprehensive tests for selective compression mode
- All 61 tests passing
- Test coverage: ~95%
- No regression in existing functionality

### Use Case
Enables incremental compression for TPC residual analysis:
- 9.6M cluster-track residuals
- 8 compressed columns
- 508 MB → 330 MB (35% file size reduction)
- Sub-micrometer precision maintained
- Compress columns incrementally as data is collected

---

##  - 2025-01-11

### Added
- Initial compression/decompression implementation
- State machine with 3 states (COMPRESSED, DECOMPRESSED, SCHEMA_ONLY)
- Bidirectional compression with mathematical transforms
- Lazy decompression via aliases
- Precision measurement (RMSE, max error, mean error)
- Schema persistence across save/load cycles
- Forward declaration support ("zero pointer" pattern)
- Collision detection for compressed column names
- ROOT TTree export with compression aliases
- Comprehensive test suite

### Features
- Compress columns using expression-based transforms
- Decompress columns with optional schema retention
- Measure compression quality metrics
- Save/load compressed DataFrames
- Export to ROOT with decompression aliases
- Recompress after modification

### Documentation
- Complete API documentation
- Usage examples
- State machine explanation


## - 2025-11-20

### Added
- **Major performance improvement**: Optimized `read_tree()` with threaded branch-by-branch reading
  - 60-770x faster read times (1s vs 771s for 12M rows)
  - 74-79% less peak memory usage
  - Default 8 worker threads (configurable via `num_workers` parameter)
- Entry range support: `entry_start` and `entry_stop` parameters for partial file reads
- Dtype restoration from `compression_info` for compressed columns

### Changed
- `read_tree()` now uses branch-by-branch reading instead of one-shot pandas construction
- Compressed columns automatically restored to correct dtype on read

### Performance
- Benchmark (12M rows, 70 branches): 771s → 1.0s
- Peak memory: 3894 MB → 853 MB
- Final DataFrame: 264 MB → 205 MB (with compression)