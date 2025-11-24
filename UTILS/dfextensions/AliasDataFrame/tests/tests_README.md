# AliasDataFrameTree Tests

## Running Tests

```bash
# Run all tests
./run_tests.sh

# Run with verbose output
./run_tests.sh -v

# Run specific test
./run_tests.sh composite_index
./run_tests.sh AliasDataFrameTree
```

## Test Files

| File | Description |
|------|-------------|
| `test_composite_index.C` | N-key composite index tests (Phase 3) |
| `test_AliasDataFrameTree.C` | Core functionality tests |

## Composite Index Tests (10 tests)

| Test | Description | Type |
|------|-------------|------|
| 1. 3-Key Semantic | Join values match formula | Correctness |
| 2. 4-Key Sparse | Real orbit values work | Infrastructure |
| 3. Merge Safety | hadd doesn't break index | I/O |
| 4. Float Detection | Non-integer rejected | Error handling |
| 5. Uniqueness | All combinations correct | Correctness |
| 6. Multiple Subframes | Different indices work | Critical |
| 7. Index Assertion | GetTreeIndex() exists | Structural |
| 8. Empty Subframe | 0-entry subframe | Edge case |
| 9. No Matches | Non-overlapping keys | Edge case |
| 10. Duplicate Keys | Consistent behavior | Edge case |

## Expected Output

```
====================================================
Composite Index Tests (with semantic verification)
====================================================
...
====================================================
Results: 10/10 tests passed
Assertions: 41 passed, 0 failed
====================================================
ALL TESTS PASSED!
```

## Writing New Tests

Use the assertion macros:

```cpp
ASSERT_TRUE(condition, "message");
ASSERT_EQ(actual, expected, "message");
ASSERT_NEAR(actual, expected, tolerance, "message");
```

Tests should:
1. Create test ROOT file with known data
2. Load via `LoadADFTree()`
3. Verify results with assertions
4. Clean up temporary files
