# GroupByRegression C++ Evaluator

**Phase 13.18.GB** — C++ port of `groupby_regression_evaluator.py` (lookup + linear methods).

Part of the ALICE TPC calibration `groupby_regression` library.
Provides fast evaluation of trained GroupByRegression models from
ROOT C++ macros, PyROOT, and (future) WASM.

## Quick Start

```bash
cd cpp

# Build the shared library (requires ROOT environment)
make libGroupByRegressionEvaluator.so

# Run Layer A tests (pure C++, no ROOT needed)
make layer_a_lookup
pytest tests/ -v -n 12

# Run Layer B smoke test (PyROOT)
python3 tests/smoke_test_layer_b.py
```

## Architecture

Two-layer design:

- **Layer A** (`gbe_kernel.hpp/.cpp`): Pure C++17 kernel. No ROOT dependency.
  WASM-compilable. Contains `GroupByRegressionEvaluator` class with
  `evaluate_lookup()` and `evaluate_linear()`.

- **Layer B** (`gbe_root_io.h/.cxx`): ROOT-dependent glue. Reads trained
  models from `.root` files via `TFile`/`TBranch`. Registers models in a
  process-static registry. Auto-generates per-model `gInterpreter->Declare`
  stubs for use in `TTree::SetAlias` and `TTree::Draw` formulas.

## Building

### Prerequisites

- g++ with C++17 support
- ROOT 6.x (tested with ROOT 6.36.04)
- Python 3.x with `numpy`, `pandas`, `uproot`, `pytest`

### Targets

```bash
make layer_a_lookup                    # Layer A cli_runner binary (tests)
make libGroupByRegressionEvaluator.so  # Shared library (Layer A + B)
make wasm_lint                         # Check Layer A for WASM-safety
make test                              # PyROOT load smoke test
make show-env                          # Print ROOT/compiler configuration
make clean                             # Remove all build artifacts
```

### Compile Flags

Layer A uses strict flags: `-std=c++17 -O2 -ffp-contract=off -Wall -Wextra
-Werror -Wshadow -Wconversion -Wsign-conversion`.

`-ffp-contract=off` is required for bit-exact parity with the Python
evaluator (suppresses FMA contraction that causes 1-ULP divergence).

Layer B uses relaxed flags (`-Wall` only) because ROOT headers trigger
`-Wshadow`/`-Wconversion` warnings.

## Usage

### From PyROOT

```python
import ROOT
ROOT.gSystem.Load("path/to/libGroupByRegressionEvaluator.so")

# Option A: load with metadata sidecar (written by dfGB_to_root.py)
ROOT.GBE.load_model_from_metadata("driftV", "models.root", "dfGB",
                                   "lookup", "nan")

# Option B: load with explicit schema
ROOT.GBE.load_model_explicit("driftV", "models.root", "dfGB",
    ROOT.std.vector("string")(["sector", "padRow"]),  # group_columns
    ROOT.std.vector("string")(["spaceCharge"]),        # predictor_columns
    ROOT.std.vector("string")(["driftV"]),             # targets
    "_fit",   # suffix
    True,     # fit_intercept
    "lookup", # method
    "nan")    # bounds

# Scalar evaluation (auto-generated stub)
prediction = ROOT.GBE.eval_driftV(5.0, 42.0, 0.3)
# Arguments: (sector, padRow, spaceCharge) as doubles
# Returns: first target's prediction, or NaN if out-of-grid

# Tabular evaluation over a TTree
f = ROOT.TFile.Open("input_data.root")
tree = f.Get("tpcData")
col_names = ROOT.std.vector("string")(["sector", "padRow", "spaceCharge"])
predictions = ROOT.GBE.eval_on_tree("driftV", tree, col_names)
# Returns std::vector<double> of length tree.GetEntries()
```

### From ROOT C++ Macros

```cpp
// Load the library
gSystem->Load("path/to/libGroupByRegressionEvaluator.so");

// Load a model (Option A: metadata sidecar)
GBE::load_model_from_metadata("driftV", "models.root", "dfGB",
                               "lookup", "nan");

// Use in TTree::SetAlias + Draw
TFile f("input_data.root");
TTree* tree = (TTree*)f.Get("tpcData");
tree->SetAlias("pred", "GBE::eval_driftV(sector, padRow, spaceCharge)");
tree->Draw("pred:driftV");  // correlation plot

// Tabular batch evaluation
auto preds = GBE::eval_on_tree("driftV", tree,
    {"sector", "padRow", "spaceCharge"});
for (int i = 0; i < 10; ++i)
    std::cout << "entry " << i << ": " << preds[i] << std::endl;
```

### Preparing Model Files

Use `dfGB_to_root.py` to dump a trained Python `dfGB` DataFrame:

```python
from dfGB_to_root import dfGB_to_root

dfGB_to_root(
    trained_dfGB,           # pandas DataFrame from make_sliding_window_fit
    "models.root",          # output file
    tree_name="dfGB",
    group_columns=["sector", "padRow"],
    predictor_columns=["spaceCharge"],
    targets=["driftV"],
    suffix="_fit",
    fit_intercept=True,
)
```

This writes a TTree + a `TObjString` metadata sidecar named
`dfGB__gbreg_schema` that `load_model_from_metadata` reads.

## API Reference

### Namespace `GBE` (Layer B)

| Function | Description |
|---|---|
| `load_model_explicit(name, path, tree, gc, pc, tgt, suffix, intercept, method, bounds)` | Load model with explicit schema |
| `load_model_from_metadata(name, path, tree, method, bounds)` | Load model from `__gbreg_schema` sidecar |
| `has_model(name)` | Check if model is registered |
| `get_model(name)` | Get pointer to evaluator (const, do not delete) |
| `list_models()` | List registered model names |
| `unload_model(name)` | Remove one model |
| `clear_models()` | Remove all models |
| `eval_on_tree(name, tree, columns)` | Batch evaluate over TTree entries |
| `eval_<name>(a0, a1, ...)` | Auto-generated per-model scalar function |

### Class `gbe::GroupByRegressionEvaluator` (Layer A)

| Method | Description |
|---|---|
| `evaluate_lookup(position_idx, predictor_values)` | Integer-position lookup |
| `evaluate_linear(position, predictor_values)` | Multilinear interpolation |
| `bin_centers()` | Per-dimension sorted natural labels |
| `remap()` | Per-dimension natural-label → compact-index map |
| `valid_mask()` | Flat bool array (true = populated bin) |
| `grid_shape()` | Per-dimension grid sizes |
| `is_valid_bin(position_idx)` | Check if a compact index is populated |

### Important: Compact Indices vs Natural Labels

All `position_idx` / `position` arguments to Layer A methods are
**compact 0..N-1 integer indices** into `bin_centers()`, NOT natural
bin labels. For example, if `sector ∈ {2, 5, 9}`, then:
- Natural label `5` → compact index `1`
- Use `remap()[dim].at(5)` to convert

The Layer B auto-generated stubs (`GBE::eval_<name>`) accept natural
labels as doubles and perform the remap internally. `eval_on_tree`
also handles the remap. Direct Layer A calls require the caller to
remap first.

## Safety Contract

- Query at a missing bin (`valid_mask = false`) → **NaN**
- Out-of-grid query with `bounds='nan'` → **NaN**
- Out-of-grid query with `bounds='clamp'` → snap to nearest edge,
  then check `valid_mask` (clamp into missing bin still returns NaN)
- **No silent neighbour fallback. No clamp-to-nearest-valid. NaN always.**

## Testing

```bash
cd cpp

# Layer A tests (43 total: 13 lookup + 15 linear + 9 safety + 6 roundtrip)
make layer_a_lookup
pytest tests/ -v -n 12

# Layer B smoke test (9 tests: load, registry, stub, tabular)
python3 tests/smoke_test_layer_b.py

# All from repo root via run_tests.sh
cd ..
bash run_tests.sh    # includes C++ tests via run_cpp_tests.sh
```

### Test Fixtures

24 deterministic JSON fixtures in `cpp/fixtures/` covering a
`fit_intercept × bounds × dimensions × method` orthogonal array
(12:12 / 12:12 / 8:8:8 / 12:12) plus 4 sparse adversarial corners.

Fixtures are committed test data (not derived artifacts). Regenerate
with `python fixtures/generate_fixtures.py` and validate with
`python fixtures/validate_fixtures.py`.

## Known Limitations (Iteration 1)

- **Methods:** `lookup` and `linear` only. `nearest`, `cubic`,
  per-dimension method dict deferred.
- **Outputs:** first target's prediction only. Multi-target and error
  column outputs (`use_errors=True`) deferred.
- **Friend-tree writing:** `eval_on_tree` returns `std::vector<double>`;
  writing back to a friend tree is a future concern.
- **WASM:** Layer A compiles to WASM (`make wasm_lint` enforces header
  discipline); actual WASM runtime integration deferred.
- **ACLiC `.C+` mode:** not yet verified. Use the `.so` for now.

## Troubleshooting

**`Error in <TCling::LoadPCM>`:** Ensure `GBE_Dict_rdict.pcm` is in the
same directory as `libGroupByRegressionEvaluator.so`. The Makefile copies
it automatically; if you move the `.so`, move the `.pcm` too.

**FMA divergence (1-ULP mismatch with Python):** Ensure `-ffp-contract=off`
is in the compile flags. Without it, g++ fuses `a*b + c*d` into FMA,
producing 1-ULP divergence vs Python's separate multiply-then-add.

**`SetBranchAddress` type errors:** The library reads group columns as
`Long64_t` and coefficient columns as `Double_t`. If your TTree uses
different types (e.g. `Int_t` for group columns), you may need to
adjust the branch types in `dfGB_to_root.py`.

**Out-of-grid returns NaN but expected a value:** Check that the query
position's natural labels exist in the model's `bin_centers()`. The
auto-generated stub remaps natural labels to compact indices; an
unrecognized label returns NaN immediately.

## File Layout

```
cpp/
├── include/
│   ├── gbe_kernel.hpp          # Layer A class header
│   └── gbe_root_io.h           # Layer B API header
├── src/
│   ├── gbe_kernel.cpp          # Layer A implementation
│   ├── gbe_root_io.cxx         # Layer B implementation
│   ├── json_reader.hpp/.cpp    # Restricted JSON parser
│   └── cli_runner.cpp          # Subprocess test binary
├── fixtures/
│   ├── generate_fixtures.py    # Deterministic fixture generator
│   ├── validate_fixtures.py    # Structural integrity checker
│   └── F_*.json                # 24 test fixtures
├── tests/
│   ├── conftest.py             # pytest path setup
│   ├── test_layer_a_lookup.py  # 13 lookup tests
│   ├── test_layer_a_linear.py  # 15 linear tests
│   ├── test_layer_a_safety.py  # 9 safety/edge-case tests
│   ├── test_dfGB_roundtrip.py  # 6 round-trip tests
│   ├── smoke_test_layer_b.py   # 9 PyROOT smoke tests
│   └── run_cpp_tests.sh        # Shell wrapper for run_tests.sh
├── dfGB_to_root.py             # Python helper: dfGB → .root
├── conftest.py                 # pytest path setup (subproject root)
├── LinkDef.h                   # ROOT dictionary directives
├── Makefile                    # Build system
├── CODER_CONTEXT.md            # Design decisions + lessons
└── README.md                   # This file
```

## References

- Phase 13.18.GB Proposal v1.1
- PHASE_13_18_GB_Fixture_Specification_v1.0
- PHASE_13_18_GBADF_v0.3 Interpolation Brainstorming
- RootInteractive (arXiv:2403.19330)
