# RDataFrameDSL Examples

Runnable examples demonstrating RDataFrameDSL capabilities.

## Prerequisites

```bash
# Ensure ROOT is available
python -c "import ROOT; print(ROOT.__version__)"

# Add RDataFrameDSL to path
export PYTHONPATH=/path/to/O2DPG/UTILS/dfextensions:$PYTHONPATH
```

## Examples

| File | Description | ROOT Required |
|------|-------------|---------------|
| `01_basic_usage.py` | Scalar expressions, arithmetic | Yes |
| `02_rvec_operations.py` | RVec indexing, slicing, masking | Yes |
| `03_method_broadcasting.py` | **Phase 8 demo** - `tracks.Pt()` | Yes |
| `04_comparison_dsl_vs_raw.py` | DSL vs raw RDataFrame side-by-side | Yes |
| `05_export_macro.py` | Export to C++ macro file | Yes |
| `06_register_function_cpp.py` | **Phase 13.5.B** - Custom C++ functions | Yes |
| `create_test_data.py` | Generate test ROOT files | Yes |

## Quick Start

```bash
# Generate test data first
python create_test_data.py

# Run any example
python 01_basic_usage.py
python 03_method_broadcasting.py  # Phase 8 demo!
python 06_register_function_cpp.py  # Phase 13.5.B demo!
```

## Demo Script (6 lines!)

The most compelling demo for physicists:

```python
from RDataFrameDSL import DSLCompiler

schema = {'tracks': 'RVec<TLorentzVector>'}
dsl = DSLCompiler(schema)
dsl.define("track_pts", "tracks.Pt()")
dsl.define("lead_pt", "tracks[:1].Pt()")
dsl.define("high_pt_eta", "tracks[tracks.Pt() > 1.0].Eta()")

print(dsl.preview())  # Show generated C++
```

This demonstrates:
- Method broadcasting (`tracks.Pt()`)
- Slice then broadcast (`tracks[:1].Pt()`)
- Filter then broadcast (`tracks[tracks.Pt() > 1.0].Eta()`)

## Custom C++ Functions (Phase 13.5.B)

Register your own C++ functions for use in RDataFrame:

```python
from RDataFrameDSL import DSLCompiler
import ROOT

dsl = DSLCompiler({"px": "double", "py": "double"})

# Register a C++ function
dsl.register_function_cpp('''
    double pt(double px, double py) {
        return sqrt(px*px + py*py);
    }
''')

# Apply to RDataFrame (declares the function)
rdf = ROOT.RDataFrame(1000)
rdf = rdf.Define("px", "gRandom->Gaus(0, 10)")
rdf = rdf.Define("py", "gRandom->Gaus(0, 10)")
rdf = dsl.apply(rdf)

# Use the function in Define
func = dsl.get_registered_function("pt")
rdf = rdf.Define("track_pt", f"{func.cpp_name}(px, py)")

print(f"Mean pT: {rdf.Mean('track_pt').GetValue():.2f}")
```

Key features:
- Hash-based naming prevents collisions (`dsl_pt_2ea43feaec756931`)
- Thread-safe declaration
- Auto-detection of headers
- FROZEN RULE #1: Lambda expressions are prohibited

## Output

Each example prints:
1. What it's demonstrating
2. The DSL expression
3. Generated C++ code (via `preview()`)
4. Execution results (when applicable)
