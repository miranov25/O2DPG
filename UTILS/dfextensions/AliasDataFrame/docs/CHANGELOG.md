# Changelog

All notable changes to AliasDataFrame will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

---

## [1.2.0] - 2025-01-15

### Added
- **Flexible missing column handling** (`on_missing` parameter)
  - Three modes: 'warn' (default), 'error', 'ignore'
  - Progressive workflow support - compress available columns, skip missing
  - Production validation mode - fail fast on missing columns
  - Silent mode for automation scripts
- **Compression summary** (`return_summary` parameter)
  - Returns dict with 'compressed' and 'skipped' column lists
  - Enables debugging and validation of compression results
  - Maintains backward compatibility (default returns self)
- **Smart column availability checking**
  - Checks DataFrame columns, aliases, AND compression_info
  - Handles compressed columns that become aliases
  - Preserves state validation for already-tracked columns
- **Uniform filtering across all modes**
  - Filtering now applies to all compression modes (including selective)
  - Consistent behavior regardless of how compress_columns is called

### Changed
- **BREAKING:** Default `on_missing` behavior changed from implicit error to 'warn'
  - Old: Missing columns raised error
  - New: Missing columns trigger warning and are skipped
  - **Migration:** Use `on_missing='error'` for strict v1.1.0 behavior
- Updated `compress_columns()` signature:
  ```python
  # Old (v1.1.0)
  compress_columns(spec=None, columns=None, suffix='_c', 
                   drop_original=True, measure_precision=False)
  
  # New (v1.2.0)
  compress_columns(spec=None, columns=None, suffix='_c', 
                   drop_original=True, 
                   on_missing='warn',        # NEW
                   return_summary=False,     # NEW
                   measure_precision=False)
  ```
- Improved error messages for missing columns
  - Shows available columns (DataFrame + aliases)
  - Provides actionable hints based on on_missing mode

### Fixed
- Selective mode now respects `on_missing` parameter
  - Previously had separate validation that always raised ValueError
  - Now uses unified filtering logic with configurable behavior
- `available_cols` and `missing_cols` now defined in all code paths
  - Previously undefined in selective mode
  - Fixed NameError in return_summary logic

### Documentation
- **NEW:** Comprehensive migration guide (v1.1.0 → v1.2.0)
- **NEW:** Progressive workflow examples
- **NEW:** Production validation patterns
- **NEW:** Troubleshooting section for missing column warnings
- Updated all examples to show new parameters
- Added best practices for each on_missing mode

### Testing
- **NEW:** 9 comprehensive tests for on_missing modes (TestCompressionOnMissing)
  - test_default_warn_mode
  - test_strict_error_mode
  - test_silent_ignore_mode
  - test_explicit_columns_subset
  - test_return_summary_default_false
  - test_all_columns_missing_warn
  - test_all_columns_missing_error
  - test_partial_missing_with_columns_param
  - test_method_chaining_still_works
- Updated test_selective_mode_validates_column_exists for new behavior
- All 70 tests passing (61 original + 9 new)
- No regression in existing functionality

### Performance
- No performance impact
- Filtering logic: O(n) where n = number of columns (negligible)
- No additional memory overhead

### Use Cases Enabled
1. **Progressive data collection**
   - Define schema for all expected columns
   - Compress available columns, skip missing (warn mode)
   - Re-run as new data arrives - automatically handles new columns
   
2. **Production validation**
   - Use error mode to ensure data quality
   - Fail fast if expected columns missing
   - Clear error messages for debugging
   
3. **Automated workflows**
   - Use ignore mode for clean logs
   - Use return_summary for monitoring
   - Track compression results programmatically

### Breaking Changes

**⚠️ Default behavior changed:**
```python
# v1.1.0
adf.compress_columns(spec)  
# → Raised error if column missing

# v1.2.0
adf.compress_columns(spec)  
# → Warns and continues if column missing

# To get v1.1.0 behavior in v1.2.0:
adf.compress_columns(spec, on_missing='error')
```

**Impact:** Low
- Most workflows benefit from flexible default
- Production code should add explicit `on_missing='error'`
- Development/incremental workflows work better with new default

**Migration checklist:**
- [ ] Review all compress_columns() calls
- [ ] Add `on_missing='error'` to production/validation code
- [ ] Consider using `return_summary=True` for debugging
- [ ] Update documentation/examples
- [ ] Test with actual data

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
  4. Selective compression: `compress_columns(spec, columns=['dy', 'dz'])`
  5. Auto-compress eligible: `compress_columns()`
- Improved error messages for compression failures
  - Specific guidance for state transition errors
  - Clear suggestions for resolution

### Testing
- Added 10 comprehensive tests for selective compression mode
- All 61 tests passing
- Test coverage: ~95%

### Use Case
Enables incremental compression for TPC residual analysis:
- 9.6M cluster-track residuals
- 8 compressed columns
- 508 MB → 330 MB (35% file size reduction)
- Sub-micrometer precision maintained

---

## [1.0.0] - 2024-XX-XX

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

---

## Version Numbering

This project uses [Semantic Versioning](https://semver.org/):
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality (backward compatible)
- **PATCH** version for bug fixes (backward compatible)

**Note:** v1.2.0 includes a breaking change to default behavior but provides easy migration path, hence MINOR version bump following common practice for "opt-in strictness" changes.

---

## Contributing

When adding entries to this changelog:
1. Add new changes to the [Unreleased] section
2. Move to versioned section on release
3. Follow the format: Added / Changed / Deprecated / Removed / Fixed / Security
4. Include use cases and examples for major changes
5. Note backward compatibility status
6. **Include migration guide for breaking changes**

---

**Last Updated:** 2025-01-15
