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
| `03_method_broadcasting.py` | **Phase 8** - `tracks.Pt()` | Yes |
| `04_comparison_dsl_vs_raw.py` | DSL vs raw RDataFrame side-by-side | Yes |
| `05_export_macro.py` | Export to C++ macro file | Yes |
| `create_test_data.py` | Generate test ROOT files | Yes |

## Quick Start

```bash
# Generate test data first
python create_test_data.py

# Run any example
python 01_basic_usage.py
python 03_method_broadcasting.py  # Phase 8 demo!
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

## Output

Each example prints:
1. What it's demonstrating
2. The DSL expression
3. Generated C++ code (via `preview()`)
4. Execution results (when applicable)

## Alias Referencing

Aliases can reference other previously defined aliases:

```python
dsl.define("pt", "sqrt(px**2 + py**2)")
dsl.define("eta", "-log(tan(atan2(pt, pz)/2))")  # Uses 'pt' alias!
dsl.define("is_central", "abs(eta) < 2.5")       # Uses 'eta' alias!
```

## Additional Features

### Visualization (Phase 12)

Generate QA plots with statistical annotations:

```python
results = dsl.draw_figures(
    specs, rdf,
    show_statistics=True,  # Add μ, σ, n stats box
    show_expected=True,    # Add N(0,1) overlay for pulls
)
```

### Export to AliasDataFrame (Phase 12)

Migrate DSL definitions to pandas workflows:

```python
schema = dsl.to_aliasdf()
adf.apply_schema(schema)
```

### Arrow Integration (Phase 13)

Export/import via PyArrow:

```python
table = dsl.to_arrow(rdf=rdf)
new_dsl = DSLCompiler.from_arrow(table)
```

---

## See Also

| Document | Description |
|----------|-------------|
| [quickstart.md](quickstart.md) | Get started in 5 minutes |
| [api_reference.md](api_reference.md) | Complete API documentation |
| [expressions.md](expressions.md) | DSL expression syntax reference |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Internal design and dataflow |
| [PHASE_HISTORY.md](PHASE_HISTORY.md) | Development history |

## Test Status

**964 tests passing, 1 skipped**
