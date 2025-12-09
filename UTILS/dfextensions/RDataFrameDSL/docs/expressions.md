# DSL Expression Reference

Complete reference for RDataFrameDSL expression syntax.

## Table of Contents

1. [Literals](#literals)
2. [Variables](#variables)
3. [Arithmetic Operations](#arithmetic-operations)
4. [Comparisons](#comparisons)
5. [Logical Operations](#logical-operations)
6. [Math Functions](#math-functions)
7. [Conditional Expressions](#conditional-expressions)
8. [Object Access](#object-access)
9. [RVec Operations](#rvec-operations)
10. [Slicing](#slicing)
11. [Method Broadcasting](#method-broadcasting)

---

## Literals

| Type | Example | C++ Output |
|------|---------|------------|
| Integer | `42` | `42` |
| Float | `3.14` | `3.14` |
| Boolean | `True`, `False` | `true`, `false` |

```python
dsl.define("answer", "42")
dsl.define("pi", "3.14159")
dsl.define("flag", "True")
```

---

## Variables

Variables reference columns from your schema:

```python
schema = {"px": "double", "pt": "RVec<double>"}
dsl = DSLCompiler(schema)

dsl.define("px_doubled", "px * 2")    # Scalar variable
dsl.define("n_tracks", "pt.size()")   # RVec variable
```

---

## Arithmetic Operations

| Operator | Example | C++ Output |
|----------|---------|------------|
| Add | `x + y` | `(x + y)` |
| Subtract | `x - y` | `(x - y)` |
| Multiply | `x * y` | `(x * y)` |
| Divide | `x / y` | `(x / y)` |
| Power | `x ** 2` | `std::pow(x, 2)` |
| Modulo | `x % 2` | `(x % 2)` (integers only) |
| Floor div | `x // y` | `std::floor((x / y))` |
| Negate | `-x` | `(-x)` |

```python
dsl.define("pt", "sqrt(px**2 + py**2)")
dsl.define("ratio", "px / py")
dsl.define("even", "n % 2")
```

---

## Comparisons

| Operator | Example | C++ Output |
|----------|---------|------------|
| Less than | `x < y` | `(x < y)` |
| Less equal | `x <= y` | `(x <= y)` |
| Greater than | `x > y` | `(x > y)` |
| Greater equal | `x >= y` | `(x >= y)` |
| Equal | `x == y` | `(x == y)` |
| Not equal | `x != y` | `(x != y)` |

```python
dsl.define("is_positive", "pt > 0")
dsl.define("is_central", "abs(eta) < 2.5")
```

---

## Logical Operations

| Operator | Example | C++ Output |
|----------|---------|------------|
| And | `x and y` | `(x && y)` |
| Or | `x or y` | `(x \|\| y)` |
| Not | `not x` | `(!x)` |

```python
dsl.define("good_track", "pt > 1.0 and abs(eta) < 2.5")
dsl.define("not_selected", "not is_good")
```

---

## Math Functions

### Standard Math (std::)

| Function | Example | Description |
|----------|---------|-------------|
| `sqrt(x)` | `sqrt(px**2 + py**2)` | Square root |
| `abs(x)` | `abs(eta)` | Absolute value |
| `sin(x)` | `sin(phi)` | Sine |
| `cos(x)` | `cos(phi)` | Cosine |
| `tan(x)` | `tan(theta)` | Tangent |
| `asin(x)` | `asin(y/r)` | Arc sine |
| `acos(x)` | `acos(z/r)` | Arc cosine |
| `atan(x)` | `atan(y/x)` | Arc tangent |
| `atan2(y, x)` | `atan2(py, px)` | 2-argument arc tangent |
| `exp(x)` | `exp(-x**2)` | Exponential |
| `log(x)` | `log(pt)` | Natural logarithm |
| `log10(x)` | `log10(energy)` | Base-10 logarithm |
| `pow(x, y)` | `pow(x, 2)` | Power (same as `x**y`) |
| `floor(x)` | `floor(x)` | Round down |
| `ceil(x)` | `ceil(x)` | Round up |
| `round(x)` | `round(x)` | Round to nearest |
| `min(x, y)` | `min(a, b)` | Minimum |
| `max(x, y)` | `max(a, b)` | Maximum |
| `hypot(x, y)` | `hypot(px, py)` | sqrt(x² + y²) |

### TMath Functions

| Function | Example | Description |
|----------|---------|-------------|
| `TMath.Gaus(x, m, s)` | `TMath.Gaus(x, 0, 1)` | Gaussian |
| `TMath.Landau(x, m, s)` | `TMath.Landau(x, 0, 1)` | Landau distribution |
| `TMath.Pi()` | `TMath.Pi()` | π constant |
| `TMath.E()` | `TMath.E()` | e constant |

```python
dsl.define("pt", "sqrt(px**2 + py**2)")
dsl.define("phi", "atan2(py, px)")
dsl.define("eta", "-log(tan(theta/2))")
```

---

## Conditional Expressions

Python-style ternary conditional:

```python
dsl.define("sign", "1 if x > 0 else -1")
dsl.define("clamped", "max_val if x > max_val else x")
```

C++ output: `(condition) ? (true_value) : (false_value)`

---

## Object Access

### Method Calls

```python
# Schema: {"particle": "TLorentzVector"}
dsl.define("pt", "particle.Pt()")
dsl.define("eta", "particle.Eta()")
dsl.define("phi", "particle.Phi()")
dsl.define("mass", "particle.M()")
```

### Property Access

```python
# Schema: {"vec": "TVector3"}
dsl.define("x", "vec.fX")
dsl.define("y", "vec.fY")
dsl.define("z", "vec.fZ")
```

### Private/Protected Members

The DSL uses TClass reflection to access private members (same as TTree::Draw):

```python
# Even protected members work via reflection
dsl.define("px", "particle.fP.fX")
```

---

## RVec Operations

### Size and Empty

```python
# Schema: {"pt": "RVec<double>"}
dsl.define("n_tracks", "pt.size()")
dsl.define("has_tracks", "not pt.empty()")
```

### Safe Indexing

**Returns NaN for out-of-bounds access (no crash):**

```python
dsl.define("first", "pt[0]")      # First element
dsl.define("last", "pt[-1]")      # Last element
dsl.define("third", "pt[2]")      # Third element (NaN if < 3 elements)
dsl.define("tenth", "pt[9]")      # NaN if fewer than 10 elements
```

### RVec Arithmetic

```python
# Element-wise operations on RVec
dsl.define("pt_scaled", "pt * 1.5")
dsl.define("pt_gev", "pt / 1000")
dsl.define("pt_sum", "px + py")  # Element-wise sum of two RVecs
```

---

## Slicing

Python-style slicing on RVec:

| Pattern | Example | Description |
|---------|---------|-------------|
| First N | `pt[:3]` | First 3 elements |
| Last N | `pt[-3:]` | Last 3 elements |
| From index | `pt[2:]` | From index 2 to end |
| Range | `pt[1:4]` | Elements 1, 2, 3 |
| Step | `pt[::2]` | Every other element |
| Reverse | `pt[::-1]` | Reversed order |
| Boolean mask | `pt[pt > 1.0]` | Elements where condition is true |

### Slice Examples

```python
dsl.define("first3", "pt[:3]")           # First 3 tracks
dsl.define("last2", "pt[-2:]")           # Last 2 tracks
dsl.define("middle", "pt[1:4]")          # Elements 1, 2, 3
dsl.define("every_other", "pt[::2]")     # Even indices
dsl.define("reversed", "pt[::-1]")       # Reversed
dsl.define("high_pt", "pt[pt > 1.0]")    # Boolean filter
```

### Safe Slicing

All slices are bounds-safe:
- `pt[:100]` returns all elements if fewer than 100 exist
- `pt[-100:]` returns all elements if fewer than 100 exist
- Empty input returns empty output

---

## Method Broadcasting

**Phase 8 Feature:** Call methods element-wise on `RVec<Object>`:

### Basic Broadcasting

```python
# Schema: {"tracks": "RVec<TLorentzVector>"}
dsl.define("track_pts", "tracks.Pt()")    # → RVec<double>
dsl.define("track_etas", "tracks.Eta()")  # → RVec<double>
dsl.define("track_phis", "tracks.Phi()")  # → RVec<double>
```

### Available Methods (TLorentzVector)

| Method | Return Type | Description |
|--------|-------------|-------------|
| `Pt()` | double | Transverse momentum |
| `Eta()` | double | Pseudorapidity |
| `Phi()` | double | Azimuthal angle |
| `M()` | double | Invariant mass |
| `E()` | double | Energy |
| `Px()` | double | X momentum |
| `Py()` | double | Y momentum |
| `Pz()` | double | Z momentum |
| `P()` | double | Total momentum |
| `Vect()` | TVector3 | 3-vector |

### Slice Then Broadcast

```python
# Get Pt of first 3 tracks
dsl.define("lead3_pt", "tracks[:3].Pt()")

# Get Eta of last 2 tracks
dsl.define("last2_eta", "tracks[-2:].Eta()")
```

### Filter Then Broadcast

```python
# Get Eta of high-pT tracks
dsl.define("high_pt_eta", "tracks[tracks.Pt() > 2.0].Eta()")
```

### Property Broadcasting

```python
# Schema: {"particles": "RVec<TParticle>"}
dsl.define("all_px", "particles.fPx")     # → RVec<double>
```

---

## Type Inference

The DSL automatically infers result types:

| Operation | Input Types | Result Type |
|-----------|-------------|-------------|
| `x + y` | double, double | double |
| `x + y` | int, double | double |
| `pt * 2.0` | RVec<double>, double | RVec<double> |
| `pt > 1.0` | RVec<double>, double | RVec<bool> |
| `tracks.Pt()` | RVec<TLorentzVector> | RVec<double> |
| `pt[:3]` | RVec<double> | RVec<double> |

---

## Error Examples

### Unknown Variable

```python
dsl.define("bad", "trakcs.Pt()")
# IRError: Unknown variable 'trakcs'
# Suggestions: Did you mean 'tracks'?
```

### Unknown Method

```python
dsl.define("bad", "tracks.Unknown()")
# IRError: Method 'Unknown' not found on element type 'TLorentzVector'
# Suggestions: Did you mean 'Pt', 'Eta', 'Phi', 'M'?
```

### Type Mismatch

```python
dsl.define("bad", "pt + 'string'")
# IRError: String constants not supported
```

---

## Generated C++ Examples

### Scalar Expression

```python
dsl.define("pt", "sqrt(px**2 + py**2)")
```

```cpp
double alias_pt(double px, double py) {
    return std::sqrt((std::pow(px, 2) + std::pow(py, 2)));
}
```

### Slicing

```python
dsl.define("first3", "pt[:3]")
```

```cpp
ROOT::RVec<double> alias_first3(const ROOT::RVec<double>& pt) {
    return [&]() -> ROOT::RVec<double> {
        size_t n = std::min(static_cast<size_t>(3), pt.size());
        return ROOT::VecOps::Take(pt, n);
    }();
}
```

### Method Broadcasting

```python
dsl.define("track_pts", "tracks.Pt()")
```

```cpp
ROOT::RVec<double> alias_track_pts(const ROOT::RVec<TLorentzVector>& tracks) {
    return [&]() -> ROOT::RVec<double> {
        ROOT::RVec<double> result;
        result.reserve(tracks.size());
        for (const auto& elem : tracks) {
            result.push_back(elem.Pt());
        }
        return result;
    }();
}
```
