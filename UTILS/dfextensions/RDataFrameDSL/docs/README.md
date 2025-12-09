# RDataFrameDSL Documentation

## What Is This?

RDataFrameDSL is a Python DSL (Domain-Specific Language) for ROOT RDataFrame that allows physicists to write analysis expressions in a Python-like syntax and have them automatically compiled to efficient C++ code.

## The Problem

Writing RDataFrame expressions in C++ is verbose and error-prone:

```cpp
// C++ - verbose, unsafe, cryptic errors
rdf.Define("track_pts", R"(
    ROOT::RVec<double> result;
    for (const auto& t : tracks) result.push_back(t.Pt());
    return result;
)");
rdf.Define("first3", "ROOT::VecOps::Take(pt, std::min((size_t)3, pt.size()))");
rdf.Define("lead_pt", "pt.size() > 0 ? pt[0] : std::nan(\"\")");
```

## The Solution

```python
# Python DSL - clean, safe, helpful errors
dsl = DSLCompiler({
    "pt": "RVec<double>",
    "tracks": "RVec<TLorentzVector>"
})
dsl.define("track_pts", "tracks.Pt()")  # Element-wise method call!
dsl.define("first3", "pt[:3]")          # Python-like slicing
dsl.define("lead_pt", "pt[0]")          # Safe indexing (returns NaN on empty)
rdf = dsl.apply(rdf)
```

## Key Features

| Feature | Example | Phase |
|---------|---------|-------|
| Scalar math | `sqrt(px**2 + py**2)` | 5 |
| Object methods | `particle.Pt()` | 6a |
| Private member access | `track.fPx` (via reflection) | 6c |
| RVec operations | `pt.size()`, `pt[0]`, `pt[-1]` | 6b |
| Python-like slicing | `pt[:3]`, `pt[-3:]`, `pt[::2]` | 7 |
| Boolean masking | `pt[pt > 1.0]` | 7 |
| **Method broadcasting** | `tracks.Pt()` → `RVec<double>` | **8 ✓** |
| **Property broadcasting** | `particles.fPx` → `RVec<double>` | **8 ✓** |
| C++ macro export | `dsl.export_macro("analysis.C")` | 7.9 |

## Current Status

- **Phases 1-8:** ✅ Complete (549 tests passing)
- **Phase 9:** Planned (RVec arithmetic type propagation)
- **Target:** ROOT team demonstration

## Quick Links

- [Architecture Overview](ARCHITECTURE.md)
- [Phase History](PHASE_HISTORY.md)
- [User Guide](USER_GUIDE.md) *(planned)*
- [Developer Guide](DEV_GUIDE.md) *(planned)*

## Repository Structure

```
RDataFrameDSL/
├── __init__.py              # Public API exports
├── ir_nodes.py              # IR node definitions
├── ir_types.py              # Type system
├── ir_errors.py             # Error types with suggestions
├── ir_builder.py            # Expression → IR
├── type_inferrer.py         # Type inference from schema
├── backend_cpp.py           # IR → C++ code generation
├── dsl_compiler.py          # High-level DSLCompiler API
└── tests/
    ├── test_ir_*.py         # IR unit tests
    ├── test_backend_*.py    # Code generation tests
    └── test_root_*.py       # ROOT integration tests
```
