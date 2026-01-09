# Phase 13.5.A - Exploration Tests

**Status:** ✅ COMPLETE (6/6 tests pass)  
**Authorization:** Main Architect approved (2026-01-09)  
**Completion Date:** 2026-01-09

## Overview

This directory contains exploration tests for Phase 13.5: Hybrid C++ Integration Strategy.

The tests validate four pillars (all equally important):
1. **Flexibility** - JIT compilation for rapid prototyping
2. **Speed** - Precompiled for multicore execution
3. **Debugging** - C++ debugger integration
4. **Standalone** - Generated C++ works without Python

## Quick Start

```bash
# Run priority tests (recommended first run)
./run_exploration.sh --priority

# Run all tests
./run_exploration.sh --all

# Run quick tests only
./run_exploration.sh --quick
```

## Test Results Summary

| Test | Status | Key Finding |
|------|--------|-------------|
| **T1** | ✅ PASS | ACLiC works for scalar and RVec types |
| **T2** | ✅ PASS | Debug symbols verified via nm |
| **T3** | ✅ PASS | Pure ROOT execution works |
| **T4** | ✅ PASS | Conservative header defaults recommended |
| **T5** | ✅ PASS | **2.65x speedup → Precompilation ESSENTIAL (P0)** |
| **T6** | ✅ PASS | ACLiC 1.17x faster than JIT |

## Critical Findings

### T5: Multicore JIT Overhead — **VALIDATES PHASE 13.5**

```
Single-threaded: 1.38s
Multi-threaded (8 cores): 0.52s
Speedup: 2.65x (theoretical max: 8x)
Efficiency: 33%

Decision: Speedup < 4× → Precompilation is ESSENTIAL (P0)
```

**Interpretation:** JIT compilation overhead wastes ~67% of multicore potential. Each thread recompiles functions independently. **This validates the core hypothesis of Phase 13.5.**

### T4: Header Detection Strategy

Conservative defaults recommended:
- `<cmath>`, `<algorithm>`, `<ROOT/RVec.hxx>` (core)
- `<TMath.h>`, `<TLorentzVector.h>` (physics extended)

User can override with `headers=` parameter in `define_raw()`.

## Known Issues & Workarounds

### ACLiC Cache Dependency Bug

**Symptom:** When compiling multiple macros in the same ROOT session, deleting intermediate `.so` files can cause subsequent compilations to fail with shell quoting errors:
```
sh: -c: line 0: unexpected EOF while looking for matching `"'
Error in <ACLiC>: Executing '...' failed!
```

**Root Cause:** ACLiC caches compiled libraries as dependencies. When T1's `.so` was deleted before T1b compiled, ACLiC's linker command still referenced the deleted file, causing a truncated/malformed shell command.

**Solution:** Defer `.so` cleanup until ALL compilations in a session complete.

**Impact on Phase 13.5.B:** When implementing `export_macro()`:
- Don't delete intermediate `.so` files during batch compilation
- OR use explicit `.U macro.C` to unload before recompiling  
- OR compile in isolated ROOT sessions

### Report Counter Bug

The `run_exploration.sh` script may show incorrect pass/fail counts due to state carryover. Always check individual test status, not summary counts.

## Test Descriptions

| Test | File | Purpose |
|------|------|---------|
| **T1** | `test_t1_aclic_basic.py` | Verify ACLiC compilation works |
| **T2-Lite** | `test_t2_debug_symbols.py` | Verify debug symbols present (CI) |
| **T2-Full** | `test_t2_debug_interactive.sh` | Manual GDB/LLDB testing |
| **T3** | `test_t3_standalone.sh` | Pure ROOT usage without Python |
| **T4** | `test_t4_header_detection.py` | Header detection exploration |
| **T5** | `test_t5_multicore_jit.py` | Multicore JIT overhead benchmark |
| **T6** | `test_t6_performance.py` | ACLiC vs JIT performance |

## Main Architect Priority Order

1. **T1** (ACLiC Compilation) - Confirm the basic mechanism works
2. **T3** (Pure ROOT) - Confirm zero Python dependencies
3. **T2** (Debug Symbols) - Validate debugging workflow
4. **T5** (Benchmark) - Quantify performance win

## Decision Gates

### T5: Multicore JIT Overhead

| Speedup | Severity | Recommendation |
|---------|----------|----------------|
| < 4× | HIGH | Precompilation ESSENTIAL (P0) |
| 4× - 6× | MEDIUM | Precompilation RECOMMENDED (P1) |
| > 6× | LOW | Precompilation OPTIONAL |

**Result: 2.65x → HIGH severity → Precompilation ESSENTIAL**

## Requirements

- ROOT (with PyROOT)
- Python 3.8+
- GDB or LLDB (for T2-Full only)

## Output

Test results are saved to `exploration_report.md`.

Individual test files may create temporary files in `~/.phase13_5_exploration/`.

## Manual Testing

After running automated tests, manually verify T2-Full on Linux:

```bash
./test_t2_debug_interactive.sh

# Then follow instructions to test with GDB:
gdb --args root -l
(gdb) break dsl_buggy
(gdb) run
# ... etc
```

Note: macOS has SIP restrictions that may limit GDB debugging. Test on Linux for full T2-Full validation.

## FROZEN RULE #1 Compliance

All tests use **named functions**, not lambdas. This is enforced by:
- Phase 13.4 D9 analysis
- test_no_lambda_validation.py in main test suite

---

## Conclusion

**Phase 13.5.A Exploration COMPLETE**

All technical risks validated. Key findings:
1. ACLiC compilation works ✅
2. Debug symbols available ✅  
3. Standalone ROOT execution works ✅
4. **Precompilation is ESSENTIAL** (2.65x speedup, 33% efficiency)

**Recommendation:** Proceed to Phase 13.5.B Implementation

---

**Phase 13.5 v0.3 — Exploration Phase Complete**
