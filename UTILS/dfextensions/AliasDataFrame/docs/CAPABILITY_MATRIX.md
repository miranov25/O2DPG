# Capability Matrix — AliasDataFrame

**Generated:** 2026-03-31 06:48 UTC
**Phase:** 13.11.ADF
**Generator:** `scripts/generate_capability_matrix.py`
**Verification:** `@pytest.mark.invariance` markers on test classes

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 13 | 4% |
| ☑️ Smoke-only | 300 | 94% |
| 🧨 Broken | 5 | 2% |
| 📋 Planned | 2 | 1% |
| **Total features** | **320** | |
| **Total tests** | **1447** | |
| **Invariance tests** | **60** | |

**Status key:**
- ✅ Verified — has `@pytest.mark.invariance` tests
- ☑️ Smoke-only — functional tests pass, no invariance verification
- 🧨 Broken — at least one test failing
- 📋 Planned — no tests yet

## Backend

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | Arrow Compute Mapper | 79 | 0 |  |
| ☑️ | Arrow Expression Edge Cases | 3 | 0 |  |
| ☑️ | Arrow Expression Evaluation | 14 | 0 |  |
| ☑️ | Arrow Expression Performance | 2 | 0 |  |
| ☑️ | Arrow Scatter Correctness | 6 | 0 |  |
| ☑️ | Arrow Scatter Integration | 2 | 0 |  |
| ☑️ | Arrow Scatter Performance | 3 | 0 |  |
| ☑️ | Arrow Scatter Primitives | 7 | 0 |  |
| ☑️ | Backend Summary | 1 | 0 |  |
| 🧨 | Backend invariance (I2) | 9 | 1 | 10 |
| ☑️ | Evaluate Expression Arrow | 3 | 0 |  |
| ☑️ | Is Arrow Available | 1 | 0 |  |
| ☑️ | Multi Column Linearization | 4 | 0 |  |
| ☑️ | Numba Availability | 2 | 0 |  |
| ☑️ | Numba Direct Accelerators | 2 | 0 |  |
| ☑️ | Numba Index Lookup | 4 | 0 |  |
| ☑️ | Numba Scatter | 4 | 0 |  |
| ☑️ | Numba With Fill Config | 1 | 0 |  |

## Compression

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | Compression Constants | 2 | 0 |  |
| ☑️ | Compression Selection | 8 | 0 |  |
| ☑️ | Compression Summary | 1 | 0 |  |
| 🧨 | Compression invariance (I4) | 6 | 2 | 8 |
| ☑️ | Describe Compression | 6 | 0 |  |
| ☑️ | Monitor Checks | 5 | 0 |  |
| ☑️ | Monitor Values | 4 | 0 |  |

## Core

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | APIContract | 6 | 0 |  |
| ☑️ | Alias Data Frame | 13 | 0 |  |
| ☑️ | Alias Data Frame Compression | 14 | 0 |  |
| ☑️ | Alias Data Frame With Subframes | 6 | 0 |  |
| ☑️ | Auto Alias Subframe Cycle Fix | 8 | 0 |  |
| ☑️ | Backward Compatibility | 3 | 0 |  |
| ☑️ | Batch Materialization Performance | 10 | 0 |  |
| ☑️ | Batch Optimization | 2 | 0 |  |
| ☑️ | Calibration Workflow | 1 | 0 |  |
| ☑️ | Column Alias Priority | 3 | 0 |  |
| ☑️ | Complex Expressions | 1 | 0 |  |
| ☑️ | Compression On Missing | 9 | 0 |  |
| ☑️ | Compression State Machine | 38 | 0 |  |
| ☑️ | Contains | 3 | 0 |  |
| ☑️ | Cycle Detection | 7 | 0 |  |
| ☑️ | Describe Structure | 6 | 0 |  |
| ☑️ | Dtype Restoration | 5 | 0 |  |
| ☑️ | Edge Cases | 10 | 0 |  |
| ☑️ | Eval In Namespace Context Override | 2 | 0 |  |
| ☑️ | Export Tree Columns | 5 | 0 |  |
| ☑️ | Fill Mode Direct | 3 | 0 |  |
| ☑️ | Fill Mode Safe | 6 | 0 |  |
| ☑️ | Get Item | 4 | 0 |  |
| ☑️ | Get Required Branches | 13 | 0 |  |
| ☑️ | Global Fill Config | 9 | 0 |  |
| ☑️ | Index Column Materialization | 4 | 0 |  |
| ☑️ | Integration With Lazy | 4 | 0 |  |
| ☑️ | Iter | 2 | 0 |  |
| ☑️ | Len | 3 | 0 |  |
| ☑️ | Materialization Behavior | 3 | 0 |  |
| ☑️ | Method Delegation | 7 | 0 |  |
| ☑️ | Mode Comparison | 1 | 0 |  |
| ☑️ | Multiple Subframes | 1 | 0 |  |
| ☑️ | Only Unmaterialized Fix | 1 | 0 |  |
| ☑️ | Parse Selection Columns | 18 | 0 |  |
| ☑️ | Parse Selection Columns Regex | 2 | 0 |  |
| ☑️ | Profiling | 8 | 0 |  |
| ☑️ | Properties | 6 | 0 |  |
| ☑️ | Read Tree Optimized | 9 | 0 |  |
| ☑️ | Read Tree With Compression | 4 | 0 |  |
| ☑️ | Resolve To Base Branches | 10 | 0 |  |
| ☑️ | Schema V2Ordering | 7 | 0 |  |
| ☑️ | Set Item Blocked | 2 | 0 |  |
| ☑️ | Subframe Fill Config | 8 | 0 |  |
| ☑️ | Warning Behavior | 4 | 0 |  |
| ☑️ | test_clean_temporary_multiple_subframes | 1 | 0 |  |
| ☑️ | test_clean_temporary_preserves_targets | 1 | 0 |  |
| ☑️ | test_clean_temporary_subframe_columns | 1 | 0 |  |
| ☑️ | test_dependency_tree_basic | 1 | 0 |  |
| ☑️ | test_dependency_tree_max_depth | 1 | 0 |  |
| ☑️ | test_dependency_tree_show_expr_false | 1 | 0 |  |
| ☑️ | test_dependency_tree_with_subframe | 1 | 0 |  |
| ☑️ | test_describe_aliases_expr_width | 1 | 0 |  |
| ☑️ | test_describe_aliases_expr_width_none | 1 | 0 |  |
| ☑️ | test_no_cleanup_when_disabled | 1 | 0 |  |
| ☑️ | test_profile_binary_output | 1 | 0 |  |
| ☑️ | test_profile_both_outputs | 1 | 0 |  |
| ☑️ | test_profile_text_output | 1 | 0 |  |

## Drawing

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | Basic | 7 | 0 |  |
| ✅ | Chain + subframe integration | 1 | 0 | 1 |
| ☑️ | Chain Lazy | 4 | 0 |  |
| ☑️ | Chain With Subframe Chain | 3 | 0 |  |
| ☑️ | Chain With Subframe Single | 2 | 0 |  |
| ✅ | Core data invariants | 6 | 0 | 6 |
| ☑️ | Draw Batch Lazy Loading | 6 | 0 |  |
| ☑️ | Draw Eager Unchanged | 3 | 0 |  |
| ☑️ | Draw Integration | 3 | 0 |  |
| ☑️ | Draw Lazy Loading | 8 | 0 |  |
| ✅ | Draw vs materialize invariance | 2 | 0 | 2 |
| ✅ | Dtype preservation | 2 | 0 | 2 |
| ☑️ | Edge Cases | 2 | 0 |  |
| ☑️ | Entry Selection | 4 | 0 |  |
| ☑️ | Error Handling | 4 | 0 |  |
| ☑️ | Error scenarios | 3 | 0 |  |
| ☑️ | File IO | 4 | 0 |  |
| ☑️ | Integration | 3 | 0 |  |
| ☑️ | Integration Errors | 3 | 0 |  |
| ☑️ | Layout | 5 | 0 |  |
| ☑️ | Lazy Loading Without Draw | 3 | 0 |  |
| ☑️ | Options | 6 | 0 |  |
| ☑️ | Performance Sanity | 2 | 0 |  |
| ☑️ | Single Eager Baseline | 4 | 0 |  |
| ☑️ | Single Lazy | 4 | 0 |  |
| ✅ | Subframe join correctness | 1 | 0 | 1 |
| ☑️ | Validation | 7 | 0 |  |
| ☑️ | With Lazy Subframe | 3 | 0 |  |
| ☑️ | draw() subframe edge cases | 4 | 0 |  |
| ☑️ | draw() subframe resolution | 10 | 0 |  |
| ☑️ | draw_batch() subframe resolution | 5 | 0 |  |
| ☑️ | draw_figures() subframe resolution | 9 | 0 |  |

## Fit Registration

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | Apply Pull Transform | 5 | 0 |  |
| ☑️ | Category Filtering | 2 | 0 |  |
| ☑️ | Compute Fit Validation | 3 | 0 |  |
| ☑️ | Draw Fit Summary Basic | 6 | 0 |  |
| ☑️ | Draw Fit Summary Options | 6 | 0 |  |
| ☑️ | Edge Cases | 4 | 0 |  |
| ☑️ | Error Surfacing | 3 | 0 |  |
| ☑️ | Fit Metadata Accessors | 5 | 0 |  |
| ☑️ | Multi Group Fit | 3 | 0 |  |
| ☑️ | Numerical Correctness | 4 | 0 |  |
| ☑️ | Plot Content | 4 | 0 |  |
| ☑️ | Register Fit Result Backward Compat | 4 | 0 |  |
| ☑️ | Register Fit Result Basic | 6 | 0 |  |
| ☑️ | Register Fit Result Formulas | 6 | 0 |  |
| ☑️ | Register Fit Result Options | 8 | 0 |  |
| ☑️ | Result Verification | 6 | 0 |  |
| ☑️ | Schema Edge Cases | 6 | 0 |  |
| ☑️ | Schema Persistence | 6 | 0 |  |
| ☑️ | Validate Fit Metadata | 8 | 0 |  |
| ☑️ | Validation Thresholds | 3 | 0 |  |

## Invariance

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ✅ | Invariance smoke (I0) | 6 | 0 | 6 |
| ☑️ | Smoke Summary | 1 | 0 |  |

## Lazy Loading

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | Auto Loading | 5 | 0 |  |
| ☑️ | Chain Creation | 11 | 0 |  |
| ☑️ | Chain Loading | 12 | 0 |  |
| ☑️ | Chain Plus Chain Integration | 1 | 0 |  |
| ☑️ | Chain Schema Consistency | 1 | 0 |  |
| ☑️ | Config Type Field | 2 | 0 |  |
| ☑️ | Edge Cases | 2 | 0 |  |
| ☑️ | Ensure Branches Method | 3 | 0 |  |
| ☑️ | Entry Selection | 5 | 0 |  |
| ☑️ | Get Subframes For Aliases | 3 | 0 |  |
| ☑️ | Laziness Preservation | 2 | 0 |  |
| ☑️ | Lazy Eager Compatibility | 2 | 0 |  |
| ☑️ | Lazy Loading Trigger | 5 | 0 |  |
| ☑️ | Lazy Properties | 6 | 0 |  |
| ☑️ | Lazy Subframe Errors | 4 | 0 |  |
| ☑️ | Lazy Subframe Join | 5 | 0 |  |
| ☑️ | Lazy Subframe Registration | 6 | 0 |  |
| ☑️ | Lazy Subframe Resources | 3 | 0 |  |
| ☑️ | Lazy Tree Reader Ensure Branches | 6 | 0 |  |
| ☑️ | Lazy Tree Reader Init | 4 | 0 |  |
| ☑️ | Lazy Tree Reader Misc | 3 | 0 |  |
| ☑️ | Load Mode Summary | 1 | 0 |  |
| ✅ | Load mode invariance (I1) | 6 | 0 | 8 |
| ☑️ | Memory Estimation | 4 | 0 |  |
| ☑️ | Parameter Validation | 3 | 0 |  |
| ☑️ | Parse Chain Files | 4 | 0 |  |
| ☑️ | Read Tree Lazy | 4 | 0 |  |
| ☑️ | Resource Management | 5 | 0 |  |
| ☑️ | Schema Consistency | 2 | 0 |  |
| ☑️ | Subframe Chain Loading | 5 | 0 |  |
| ☑️ | Subframe Chain Registration | 6 | 0 |  |
| ☑️ | Subframe Chain Resources | 2 | 0 |  |
| ☑️ | Subframe Chain Validation | 2 | 0 |  |
| ☑️ | Validation Modes | 10 | 0 |  |

## Other

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ✅ | Fill Value Dependency Resolution | 7 | 0 | 1 |

## RDataFrame

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| 🧨 | Add Defines Collision | 4 | 1 |  |
| ☑️ | Add Defines To RDF | 3 | 0 |  |
| ☑️ | Cache To Snapshot | 1 | 0 |  |
| ☑️ | Chain Setup | 2 | 0 |  |
| ☑️ | Composite Key Helpers | 7 | 0 |  |
| ☑️ | Compute Composite Key Auto | 3 | 0 |  |
| ☑️ | Compute Composite Key Dense | 4 | 0 |  |
| ☑️ | Compute Composite Key Sparse | 3 | 0 |  |
| ☑️ | Extract Dependencies | 5 | 0 |  |
| ☑️ | Generate Rdf Code | 3 | 0 |  |
| ☑️ | Get Ordered Defines | 5 | 0 |  |
| ☑️ | Join Columns For Snapshot | 1 | 0 |  |
| ☑️ | Modular RDFSetup | 4 | 0 |  |
| ✅ | Ordered Defines With Schema | 10 | 0 |  |
| ✅ | RDFIntegration | 12 | 0 |  |
| 📋 | RDFMulti Thread | 0 | 0 |  |
| 📋 | RDFReal Data | 0 | 0 |  |
| ☑️ | RData Frame Basics | 1 | 0 |  |
| 🧨 | RData Frame Friend Access | 3 | 1 |  |
| ☑️ | RData Frame Index Verification | 3 | 0 |  |
| ☑️ | RData Frame With Real Schema | 1 | 0 |  |
| ☑️ | Runtime Composite Key | 5 | 0 |  |
| ☑️ | Should Use Sparse | 3 | 0 |  |
| 🧨 | TMem File Branch | 10 | 1 |  |
| ☑️ | TTree Draw Multi Key Subframe | 2 | 0 |  |
| ☑️ | TTree Draw Strict Accuracy | 1 | 0 |  |
| ☑️ | TTree Draw Subframe | 7 | 0 |  |
| ☑️ | To Cpp Expr | 22 | 0 |  |

## Registered Functions

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | PolynomialSpec — ROOT expression | 2 | 0 |  |
| ☑️ | PolynomialSpec — basis expressions | 9 | 0 |  |
| ✅ | PolynomialSpec — invariance | 3 | 0 | 3 |
| ☑️ | PolynomialSpec — schema roundtrip | 5 | 0 |  |
| ☑️ | Repr | 2 | 0 |  |
| ☑️ | register_evaluator — basic | 4 | 0 |  |
| ☑️ | register_evaluator — collision/overwrite | 3 | 0 |  |
| ☑️ | register_evaluator — composition | 4 | 0 |  |
| ☑️ | register_evaluator — edge cases | 4 | 0 |  |
| ✅ | register_evaluator — invariance | 3 | 0 | 3 |
| ☑️ | register_evaluator — multi-predictor | 4 | 0 |  |
| ☑️ | register_evaluator — validation | 2 | 0 |  |
| ☑️ | register_function | 4 | 0 |  |
| ☑️ | register_polynomial_from_subframe | 1 | 0 |  |

## Schema

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | Add Alias Schema | 8 | 0 |  |
| ☑️ | Add Validation Indicator | 2 | 0 |  |
| ☑️ | Add Validation Summary | 2 | 0 |  |
| ☑️ | Alias Dtypes Property | 12 | 0 |  |
| ☑️ | Aliases Property | 16 | 0 |  |
| ☑️ | Apply Aliases | 8 | 0 |  |
| ☑️ | Apply Dtypes | 12 | 0 |  |
| ☑️ | Apply Schema | 2 | 0 |  |
| ☑️ | Column Ordering By Groups | 5 | 0 |  |
| ☑️ | Column Spec Export | 6 | 0 |  |
| ☑️ | Compression Info Property | 10 | 0 |  |
| ☑️ | Compute Statistics | 4 | 0 |  |
| ☑️ | Constant Aliases Property | 12 | 0 |  |
| ☑️ | Convert Dtypes | 2 | 0 |  |
| ☑️ | Cycle Detection | 3 | 0 |  |
| ☑️ | Definition Vs Record Export | 8 | 0 |  |
| ☑️ | Deprecation Warning | 3 | 0 |  |
| ☑️ | Describe Data | 3 | 0 |  |
| ☑️ | Describe Schema | 2 | 0 |  |
| ☑️ | Draw Fit Summary Integration | 6 | 0 |  |
| ☑️ | Edge Cases | 8 | 0 |  |
| ☑️ | From Schema | 1 | 0 |  |
| ☑️ | Full Workflow | 2 | 0 |  |
| ☑️ | Index Column Repair | 4 | 0 |  |
| ☑️ | Jq Queryability | 3 | 0 |  |
| ☑️ | Mutation Safety | 10 | 0 |  |
| ☑️ | PolynomialSpec — schema roundtrip | 4 | 0 |  |
| ☑️ | Register Subframe Schema | 4 | 0 |  |
| ☑️ | Schema Change Protection | 3 | 0 |  |
| ☑️ | Schema Integration | 6 | 0 |  |
| ☑️ | Schema Loading | 4 | 0 |  |
| ☑️ | Schema Round Trip | 1 | 0 |  |
| ☑️ | Schema Serialization | 2 | 0 |  |
| ☑️ | Schema Structure | 8 | 0 |  |
| ☑️ | Select Data | 7 | 0 |  |
| ☑️ | Select Schema | 2 | 0 |  |
| ☑️ | Smart Json Formatting | 4 | 0 |  |
| ☑️ | Subframe Alias Vs Materialized | 2 | 0 |  |
| ☑️ | Subframe Registry Has Subframe | 4 | 0 |  |
| ☑️ | Subframe Schema Population | 7 | 0 |  |
| ☑️ | Update Schema | 20 | 0 |  |
| ☑️ | Validate Schema Check Data | 3 | 0 |  |
| ☑️ | Validate Schema Strict | 5 | 0 |  |
| ☑️ | Validate Schema Update | 6 | 0 |  |
| ☑️ | Validation Output Structure | 4 | 0 |  |

## Subframes

| Status | Feature | Passed | Failed | Invariance |
|--------|---------|-------:|-------:|:----------:|
| ☑️ | Alias Data Frame RDFReexports | 2 | 0 |  |
| ☑️ | Check Dense Overflow | 5 | 0 |  |
| ☑️ | Compute Composite Key Auto | 5 | 0 |  |
| ☑️ | Compute Composite Key Dense | 5 | 0 |  |
| ☑️ | Compute Composite Key Sparse | 4 | 0 |  |
| ☑️ | Generate Dense Cpp Expression | 5 | 0 |  |
| ☑️ | Get Composite Key Column Name | 4 | 0 |  |
| ☑️ | Join Caching Correctness | 5 | 0 |  |
| ☑️ | Join Caching Edge Cases | 3 | 0 |  |
| ☑️ | Join Caching Lifecycle | 3 | 0 |  |
| ☑️ | Join Caching Performance | 2 | 0 |  |
| ☑️ | Join Index Caching | 9 | 0 |  |
| ☑️ | Module Exports | 2 | 0 |  |
| ☑️ | Multi Key Subframe Joins | 3 | 0 |  |
| ☑️ | Should Use Sparse | 4 | 0 |  |
| ☑️ | Subframe Basic Join | 4 | 0 |  |
| ☑️ | Subframe Edge Cases | 5 | 0 |  |
| ☑️ | Subframe Index Materialization | 4 | 0 |  |
| ☑️ | Subframe Lazy Evaluation | 4 | 0 |  |
| ☑️ | Subframe Missing Keys | 5 | 0 |  |
| ☑️ | Subframe Roundtrip | 3 | 0 |  |
| ☑️ | Subframe Summary | 1 | 0 |  |
| ☑️ | Subframe Vs Flattened | 1 | 0 |  |
| ✅ | Subframe join invariance (I3) | 9 | 0 | 9 |
| ☑️ | test_add_alias_basic | 1 | 0 |  |
| ☑️ | test_alias_chain | 1 | 0 |  |
| ☑️ | test_alias_lifecycle | 1 | 0 |  |
| ☑️ | test_alias_name_collision | 1 | 0 |  |
| ☑️ | test_alias_with_complex_expression | 1 | 0 |  |
| ☑️ | test_auto_alias_tracking_manual | 1 | 0 |  |
| ☑️ | test_auto_aliases_dict_exists | 1 | 0 |  |
| ☑️ | test_backward_compatibility_no_auto_aliases | 1 | 0 |  |
| ☑️ | test_concurrent_modifications | 1 | 0 |  |
| ☑️ | test_empty_alias_dict_operations | 1 | 0 |  |
| ☑️ | test_get_auto_aliases_empty | 1 | 0 |  |
| ☑️ | test_get_auto_aliases_filtered | 1 | 0 |  |
| ☑️ | test_indirect_subframe_reference | 1 | 0 |  |
| ☑️ | test_is_auto_alias_false_for_manual | 1 | 0 |  |
| ☑️ | test_large_number_of_aliases | 1 | 0 |  |
| ☑️ | test_list_auto_aliases_empty | 1 | 0 |  |
| ☑️ | test_list_auto_aliases_filtered | 1 | 0 |  |
| ☑️ | test_list_subframes_empty | 1 | 0 |  |
| ☑️ | test_materialize_alias_works | 1 | 0 |  |
| ☑️ | test_multi_subframe_auto_aliasing | 1 | 0 |  |
| ☑️ | test_multi_subframe_column_collision | 1 | 0 |  |
| ☑️ | test_multiple_alias_operations | 1 | 0 |  |
| ☑️ | test_remove_alias_basic | 1 | 0 |  |
| ☑️ | test_remove_alias_from_auto_aliases | 1 | 0 |  |
| ☑️ | test_remove_alias_keep_in_schema | 1 | 0 |  |
| ☑️ | test_remove_alias_lenient_no_error | 1 | 0 |  |
| ☑️ | test_remove_alias_schema_sync | 1 | 0 |  |
| ☑️ | test_remove_alias_strict_error | 1 | 0 |  |
| ☑️ | test_remove_aliases_multiple | 1 | 0 |  |
| ☑️ | test_remove_auto_aliases_all | 1 | 0 |  |
| ☑️ | test_remove_auto_aliases_empty | 1 | 0 |  |
| ☑️ | test_remove_auto_aliases_selective | 1 | 0 |  |
| ☑️ | test_schema_embedding_root_roundtrip | 1 | 0 |  |
| ☑️ | test_schema_export_import_basic | 1 | 0 |  |
| ☑️ | test_schema_roundtrip_preserves_auto_alias_info | 1 | 0 |  |
| ☑️ | test_special_characters_in_alias_name | 1 | 0 |  |
| ☑️ | test_subframe_self_reference_regression | 1 | 0 |  |

## 🧨 Broken Features — Details

### Add Defines Collision

- ❌ `dfextensions/AliasDataFrame/tests/test_AliasDataFrameRDF.py::TestAddDefinesCollision::test_collision_from_friend_tree`

### Backend invariance (I2)

- ❌ `dfextensions/AliasDataFrame/tests/test_invariance_backend.py::TestInvarianceBackend::test_I2_6_chained_subframe_expressions_numba_vs_numpy`

### Compression invariance (I4)

- ❌ `dfextensions/AliasDataFrame/tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_2_scaled_linear_compression_roundtrip`
- ❌ `dfextensions/AliasDataFrame/tests/test_invariance_compression.py::TestInvarianceCompression::test_I4_3_asinh_compression_roundtrip`

### TMem File Branch

- ❌ `dfextensions/AliasDataFrame/tests/test_AliasDataFrameRDF.py::TestTMemFileBranch::test_missing_keys_in_friend`

### RData Frame Friend Access

- ❌ `dfextensions/AliasDataFrame/tests/test_AliasDataFrameRDF.py::TestRDataFrameFriendAccess::test_composite_index_friend`

---

*Auto-generated from pytest results. ✅ = @pytest.mark.invariance test exists. ☑️ = smoke tests only.*