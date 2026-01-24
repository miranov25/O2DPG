# Schema Validation Audit Summary

**Document ID:** SCHEMA_VALIDATION_AUDIT_v1.2.md  
**Author:** Claude-Opus-4.5 (Coder Reviewer)  
**Date:** 2026-01-24  
**Purpose:** Pre-implementation audit for Phase 13.6.F Layer 1 + alias() proposal  
**Status:** APPROVED with corrections (per 7-reviewer consensus)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| v1.0 | 2026-01-23 | Initial audit |
| v1.1 | 2026-01-23 | Added `alias()` proposal (Section 7), TTree::SetAlias compatibility |
| v1.2 | 2026-01-24 | P0 fixes: corrected to_pandas() validation claims, added pool-based model, added design contracts (Q1-Q4), marked Section 7 as PROPOSED EXTENSION |

---

## 1. Executive Summary

This audit documents how RDataFrameDSL currently handles schema and validation, identifying what exists vs. what Phase 13.6.F needs to add.

**Key Finding:** The existing infrastructure is more complete than expected. Layer 1 is primarily enhancement, not new implementation.

**Key Extension:** Section 7 proposes `alias()` method for TTree::SetAlias-style deferred validation with pool-based compilation.

---

## 2. Schema Propagation Flow

### 2.1 Current Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SCHEMA SOURCES                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Option A: Manual Schema              Option B: Auto-Infer from Tree │
│  ─────────────────────                ───────────────────────────── │
│  schema = {                           dsl = DSLCompiler.from_tree(   │
│      'px': 'double',                      'data.root',               │
│      'py': 'double',                      'Events'                   │
│  }                                    )                              │
│  dsl = DSLCompiler(schema)            # Schema auto-extracted        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     TYPE INFERRER                                    │
├─────────────────────────────────────────────────────────────────────┤
│  TypeInferrer.from_schema(schema)  OR  TypeInferrer.from_tree(tree) │
│                                                                      │
│  Stores:                                                             │
│  - _variables: Dict[str, VariableInfo]  # Column metadata            │
│  - _alias_types: Dict[str, VariableInfo]  # Computed column types    │
│  - _schema: Original schema dict                                     │
│                                                                      │
│  Provides:                                                           │
│  - get_variable_info(name) → VariableInfo or raises IRError          │
│  - _find_similar_names(name) → List[str] for suggestions             │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     IR BUILDER                                       │
├─────────────────────────────────────────────────────────────────────┤
│  IRBuilder(inferrer)                                                 │
│                                                                      │
│  On build(expression):                                               │
│  1. Parse Python AST                                                 │
│  2. For each variable: inferrer.get_variable_info(name)              │
│     → If not found: raises IRError with suggestions                  │
│  3. Type-check operations                                            │
│  4. Return IR tree                                                   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DSL COMPILER                                     │
├─────────────────────────────────────────────────────────────────────┤
│  define(name, expression):                                           │
│  1. Check name collision with schema                                 │
│  2. IRBuilder.build(expression)  ← VALIDATION HAPPENS HERE           │
│  3. Generate C++ code                                                │
│  4. Register alias type for future expressions                       │
│                                                                      │
│  apply(rdf):                                                         │
│  1. Compile all functions                                            │
│  2. rdf.Define() for each definition                                 │
│                                                                      │
│  to_pandas(rdf, columns):                                            │
│  1. apply(rdf) ← applies definitions                                 │
│  2. AsNumpy(columns) ← ROOT validates column existence               │
│  3. Flatten to DataFrame                                             │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Validation Timeline

| Operation | Validation Performed | Error Type |
|-----------|---------------------|------------|
| `DSLCompiler(schema)` | Schema format check | ValueError |
| `dsl.define(name, expr)` | Column existence, type checking, method validation | IRError |
| `dsl.apply(rdf)` | C++ compilation | ROOT exception |
| `dsl.to_pandas(rdf, cols)` | Calls apply(), then AsNumpy() - ROOT validates columns | ROOT exception |

**P0 CORRECTION (v1.2):** `to_pandas()` does NOT use `GetColumnNames()`/`GetColumnType()` for validation. It calls `apply()` then `AsNumpy()`. Column validation happens in ROOT's AsNumpy() call.

**Code evidence (dsl_compiler.py:2066-2082):**
```python
def to_pandas(self, rdf, columns, ...):
    # Apply DSL definitions first
    applied_rdf = self.apply(rdf)
    # ...
    # Get data from RDataFrame - ROOT validates here
    data = applied_rdf.AsNumpy(columns_to_fetch)
```

**Key Insight:** Validation happens at **define-time** in `IRBuilder._visit_Name()` (line 678):

```python
# ir_builder.py:678-684
if not self.inferrer.has_variable(name, ctx.namespace):
    similar = self.inferrer._find_similar_names(name)
    raise unknown_variable_error(
        name, 
        location=self._make_location(node, ctx),
        similar_names=similar
    )
```

### 2.3 Schema Update After Define (IMPORTANT)

After each `define()`, the schema is **updated** to include the new alias:

```python
# dsl_compiler.py:544-545 (in define())
# === NEW: Register alias in schema for future expressions ===
self._register_alias_type(name, ir, dtype)

# dsl_compiler.py:647-652 (in _register_alias_type())
# Add to simple schema
self.schema[name] = type_str

# Rebuild TypeInferrer with updated schema
full_schema = _simple_schema_to_full(self.schema)
self._inferrer = TypeInferrer.from_schema(full_schema)
```

**This means:** Alias chaining works because each `define()` adds to schema before next `define()` validates.

---

## 3. Current Schema Options

### 3.1 Option A: Manual Schema (Current Primary)

```python
schema = {
    'px': 'double',
    'py': 'double', 
    'tracks': 'RVec<TLorentzVector>',
}
dsl = DSLCompiler(schema)
```

**Pros:** No ROOT required at construction, testable  
**Cons:** User must know types, can be out of sync with data

### 3.2 Option B: Auto-Infer from TTree

```python
dsl = DSLCompiler.from_tree('data.root', 'Events')
# Schema auto-extracted from tree branches
```

**Implementation:** `dsl_compiler.py:1520-1599`

**Pros:** Automatic, always in sync  
**Cons:** Requires ROOT, file must exist

### 3.3 Option C: From RDataFrame (NOT IMPLEMENTED)

```python
# THIS DOES NOT EXIST YET
rdf = ROOT.RDataFrame('Events', 'data.root')
dsl = DSLCompiler.from_rdf(rdf)  # ← Would use GetColumnNames/GetColumnType
```

**Status:** Not implemented. Would enable schema-less workflow.

**Note (v1.2):** `GetColumnType()` may fail for some complex template types. Implementation should handle gracefully.

---

## 4. Error Handling (Existing)

### 4.1 Unknown Variable Error

**File:** `ir_errors.py:355-371`

```python
def unknown_variable_error(
    name: str,
    location: Optional[SourceLocation] = None,
    similar_names: List[str] = None
) -> IRError:
    """Create an unknown variable error with suggestions."""
    suggestions = []
    if similar_names:
        for similar in similar_names[:3]:
            suggestions.append(f"Did you mean '{similar}'?")
    
    return IRError(
        kind=IRErrorKind.TYPE_ERROR,
        message=f"Unknown variable '{name}' - not found in tree or aliases",
        source_location=location,
        suggestions=suggestions
    )
```

### 4.2 Similar Name Finding

**File:** `type_inferrer.py:836-855`

```python
def _find_similar_names(self, target: str, max_results: int = 3) -> List[str]:
    """Find similar variable names for error suggestions."""
    target_lower = target.lower()
    all_names = self.get_column_names()
    
    matches = []
    for name in all_names:
        name_lower = name.lower()
        # Check substring match (both directions)
        if target_lower in name_lower or name_lower in target_lower:
            matches.append(name)
        # Check prefix match
        elif name_lower.startswith(target_lower[:3]) if len(target_lower) >= 3 else False:
            matches.append(name)
    
    return matches[:max_results]
```

**Note (v1.2):** Current algorithm (substring/prefix matching) provides useful suggestions for common typos. Phase 13.6.F will upgrade to `difflib.get_close_matches()` with threshold 0.6 for improved fuzzy matching.

### 4.3 Test Coverage

| Test | File | Line |
|------|------|------|
| `test_unknown_variable_error` | test_ir_builder.py | 152 |
| `test_unknown_variable_suggestions` | test_ir_builder.py | 158 |
| `test_unknown_function` | test_ir_builder.py | 435 |

---

## 5. What Exists vs. What's Needed

### 5.1 ✅ Already Implemented

| Feature | Location | Notes |
|---------|----------|-------|
| Column existence check | `TypeInferrer.get_variable_info()` | Raises IRError |
| Type/rank tracking | `TypeInferrer`, `IRBuilder` | Full type system |
| Suggestions for typos | `_find_similar_names()` | Substring matching |
| "Did you mean X?" errors | `unknown_variable_error()` | Up to 3 suggestions |
| Method validation | `IRBuilder` + ROOT reflection | Via TClass |
| Schema from TTree | `DSLCompiler.from_tree()` | Auto-infer |
| Alias chaining | `define()` registers alias types | Aliases can use aliases |

### 5.2 ❌ Gaps for Phase 13.6.F

| Gap | Current | Needed | Effort |
|-----|---------|--------|--------|
| Suggestion algorithm | Substring/prefix | difflib (threshold 0.6) | 10 lines |
| Error format | Basic message | "Available columns: [list]" | 5 lines |
| DSLError alias | Only IRError | DSLError = IRError | 1 line |
| From RDataFrame | Not implemented | `DSLCompiler.from_rdf(rdf)` | 30 lines |
| Deferred validation | Validate at define-time only | `alias()` method | See Section 7 |
| batch_draw() | Not implemented | New API | Per proposal |

---

## 6. Schema-Less Operation (Design Question)

### 6.1 TTree::SetAlias Comparison

In ROOT's `TTree::SetAlias`, aliases work **without schema**:

```cpp
// ROOT TTree - no schema needed, validation at Draw() time
tree->SetAlias("pt", "sqrt(px*px + py*py)");
tree->SetAlias("high_pt", "pt > 10");  // Uses 'pt' alias
tree->SetAlias("eta", "track_eta[0]");  // References unknown column
tree->Draw("high_pt");  // Validation happens HERE at execution
```

**Key difference:** TTree validates at **Draw-time** (execution), not at SetAlias-time (definition).

### 6.2 Current RDataFrameDSL Limitation

DSLCompiler **requires** schema at construction:

```python
def __init__(self, schema: Dict[str, str], safe_indexing: bool = True):
    # schema is required, not optional
```

Schema is checked at **define-time** in `IRBuilder._visit_Name()`:
```python
if not self.inferrer.has_variable(name, ctx.namespace):
    raise unknown_variable_error(name, ...)
```

### 6.3 Desired "Easy Definition" Mode

Main Architect's intent:
> "In the old TTree::SetAlias we could define aliases without knowing schema."
> "We should be able to update schema from_rdf. Later we can make schema continuation."

### 6.4 Proposed Solutions

**Option 1: from_rdf() - Infer Schema at Construction**

```python
@classmethod
def from_rdf(cls, rdf, safe_indexing: bool = True) -> 'DSLCompiler':
    """Create DSLCompiler with schema auto-inferred from RDataFrame."""
    schema = {}
    for col in rdf.GetColumnNames():
        col_name = str(col)
        col_type = str(rdf.GetColumnType(col_name))
        schema[col_name] = col_type
    
    instance = cls(schema, safe_indexing=safe_indexing)
    instance._rdf = rdf  # Store reference
    return instance
```

**Note:** `GetColumnType()` may return complex template strings or fail for some types. Implementation should handle gracefully with fallback to "Unknown".

**Option 2: Deferred Validation (TTree::SetAlias Style)** → See Section 7 (`alias()` method)

**Option 3: update_schema_from_rdf() Method**

```python
def update_schema_from_rdf(self, rdf) -> 'DSLCompiler':
    """
    Update schema with columns from RDataFrame.
    
    Allows incremental schema building:
    1. Start with partial schema (or empty)
    2. Update from RDF when available
    3. Define expressions with full schema
    
    Schema merging rule: Manual schema entries take precedence
    over RDF-inferred entries (don't overwrite existing).
    """
    for col in rdf.GetColumnNames():
        col_name = str(col)
        if col_name not in self.schema:  # Don't overwrite manual entries
            col_type = str(rdf.GetColumnType(col_name))
            self.schema[col_name] = col_type
    
    # Rebuild inferrer with updated schema
    full_schema = _simple_schema_to_full(self.schema)
    self._inferrer = TypeInferrer.from_schema(full_schema)
    
    return self
```

### 6.5 Recommendation

**Phase 13.6.F:** Implement **Option 1 (from_rdf)** + **Option 3 (update_schema_from_rdf)** + **Option 2 (alias())**

- `from_rdf()`: Simple, consistent with `from_tree()` pattern
- `update_schema_from_rdf()`: Enables incremental workflow
- `alias()`: Full TTree::SetAlias compatibility (see Section 7)

---

## 7. PROPOSED EXTENSION: `alias()` Method - Pool-Based Deferred Validation

> **Note (v1.2):** This section describes a PROPOSED EXTENSION, not current functionality.
> Implementation is planned for Phase 13.6.F.

### 7.1 Motivation: TTree::SetAlias Compatibility

In ROOT's TTree, aliases work without upfront schema:

```cpp
// ROOT TTree - no schema needed, validation at Draw() time
tree->SetAlias("pt", "sqrt(px*px + py*py)");
tree->SetAlias("high_pt", "pt > 10");  // References 'pt' alias
tree->SetAlias("eta", "track_eta[0]");  // References unknown column
tree->Draw("high_pt");  // Validation happens HERE
```

**Key insight:** Formulas are stored first, validated only when needed.

### 7.2 Proposed API

```python
class DSLCompiler:
    def define(self, name: str, expression: str) -> 'DSLCompiler':
        """
        Define column with IMMEDIATE validation (current behavior).
        
        Requires schema. Validates against schema at call time.
        """
        ...
    
    def alias(self, name: str, expression: str) -> 'DSLCompiler':
        """
        Define alias with DEFERRED validation (TTree::SetAlias style).
        
        Formula stored but NOT validated until apply()/draw()/to_pandas().
        Allows referencing columns not yet in schema.
        """
        self._aliases[name] = expression  # Raw storage only
        return self
```

### 7.3 Semantics Comparison

| Method | Validation | Schema Required | ROOT Equivalent |
|--------|------------|-----------------|-----------------|
| `define()` | Immediate (at call) | Yes | `rdf.Define()` |
| `alias()` | Deferred (at `apply()`/`draw()`) | No | `tree->SetAlias()` |

### 7.4 Pool-Based Compilation Model

**CRITICAL DESIGN DECISION (v1.2):**

The `alias()` method uses a **POOL-BASED** model, not linear processing:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     POOL-BASED ALIAS MODEL                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ALIAS POOL (template library)         REQUESTED          COMPILED   │
│  ─────────────────────────────         ─────────          ────────  │
│  dsl.alias("pt", "sqrt(px**2+py**2)")                               │
│  dsl.alias("eta", "-log(tan(θ/2))")    User calls:        Only      │
│  dsl.alias("phi", "atan2(py,px)")      draw("high_pt")    needed    │
│  dsl.alias("high_pt", "pt > 10")       ───────────────►   aliases   │
│  dsl.alias("low_eta", "abs(eta)<1")                       compiled  │
│  dsl.alias("unused1", "...")           Traces deps:                  │
│  dsl.alias("unused2", "...")           high_pt → pt                  │
│                                        pt → px, py                   │
│                                                                      │
│  Result: Only "pt" and "high_pt" validated/compiled                 │
│          "eta", "phi", "low_eta", "unused1", "unused2" IGNORED      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this matters:**
- Enables **template libraries** with many pre-defined aliases
- User loads library once, uses subset per analysis
- Unused aliases never validated → no spurious errors
- Matches TTree::SetAlias mental model exactly

### 7.5 Dependency Resolution

**Algorithm:** Simple recursive dependency tracing (not topological sort):

```python
def _get_needed_aliases(self, requested: List[str]) -> Set[str]:
    """
    Trace dependencies backward from requested columns.
    Returns set of aliases that need compilation.
    """
    needed = set()
    visited = set()  # For cycle detection
    
    def trace(name):
        if name in visited:
            # Cycle detected
            raise IRError(IRErrorKind.CYCLE_ERROR, 
                f"Circular dependency detected involving '{name}'")
        if name in self._aliases and name not in needed:
            visited.add(name)
            needed.add(name)
            for dep in self._extract_dependencies(self._aliases[name]):
                trace(dep)
            visited.remove(name)
    
    for name in requested:
        trace(name)
    return needed
```

### 7.6 Design Contracts (Q1-Q4 from Review)

**Q1: Materialization Contract**

> When `dsl.draw("high_pt", rdf)` is called, does it materialize aliases only for requested expressions, or all registered aliases?

**ANSWER: GUARANTEED CONTRACT** - Only requested columns and their dependencies are validated/compiled. Unused aliases are **never touched**.

**Q2: Cycle Detection**

> What is the behavior for cycles like `a = b + 1`, `b = a + 1`?

**ANSWER: REQUIRED** - Cycle detection via recursion stack. Returns `IRError(CYCLE_ERROR)` with message identifying the cycle.

| Case | Behavior |
|------|----------|
| `a → b → a` | `IRError(CYCLE_ERROR, "Circular dependency detected involving 'a'")` |
| `a → a` (self) | `IRError(CYCLE_ERROR, "Circular dependency detected involving 'a'")` |

**Q3: Name Resolution Order**

> For a token `x` in an alias expression, what is the lookup precedence?

**ANSWER: DEFINED ORDER**

1. `define()` columns (already validated, in schema)
2. Alias pool (deferred, validate now)
3. RDF columns via `GetColumnNames()`/`GetColumnType()`
4. Error with suggestions ("Did you mean...?" + "Available columns: [list]")

**Q4: Type Assignment**

> How are types assigned for RDF-derived columns and alias outputs?

**ANSWER:**
- RDF columns: Type read via `rdf.GetColumnType(col)`, stored in schema verbatim
- Alias outputs: After compilation, output type registered in schema for downstream aliases
- Constraints: Unknown template types stored as "Unknown", validated at C++ compile time

### 7.7 Validation Timeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DEFINE vs ALIAS TIMELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  dsl.define("pt", "sqrt(px**2 + py**2)")                            │
│       │                                                              │
│       ▼                                                              │
│  [IMMEDIATE VALIDATION]                                              │
│  - Parse expression                                                  │
│  - Check px, py in schema ← ERROR if missing                        │
│  - Type check                                                        │
│  - Generate C++ code                                                 │
│  - Add 'pt' to schema                                                │
│                                                                      │
│  dsl.alias("high_pt", "pt > 10")                                    │
│       │                                                              │
│       ▼                                                              │
│  [STORE ONLY]                                                        │
│  - Save expression string to pool                                    │
│  - NO validation                                                     │
│  - NO schema check                                                   │
│                                                                      │
│  dsl.draw("high_pt", rdf)  OR  dsl.apply(rdf)  OR  dsl.to_pandas()  │
│       │                                                              │
│       ▼                                                              │
│  [DEFERRED VALIDATION - POOL-BASED]                                  │
│  1. Determine requested columns                                      │
│  2. Trace dependencies: high_pt → pt → px, py                       │
│  3. Update schema from RDF (GetColumnNames/GetColumnType)            │
│  4. Merge schema (manual > RDF, don't overwrite)                    │
│  5. Validate ONLY needed aliases (cycle detection)                   │
│  6. Compile ONLY needed aliases                                      │
│  7. Register output types for downstream                             │
│  8. Apply to RDF                                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.8 Usage Examples

```python
# Example 1: Schema-based workflow (current, unchanged)
dsl = DSLCompiler({'px': 'double', 'py': 'double'})
dsl.define("pt", "sqrt(px**2 + py**2)")  # ✅ Validated immediately

# Example 2: SetAlias-style workflow (new)
dsl = DSLCompiler({})  # Empty schema OK
dsl.alias("pt", "sqrt(px**2 + py**2)")   # Stored, not validated
dsl.alias("high_pt", "pt > 10")          # Stored, not validated
rdf = dsl.apply(rdf)  # NOW: schema inferred, needed aliases validated

# Example 3: Template library workflow
dsl = DSLCompiler({})
# Load template library (many aliases)
dsl.alias("pt", "sqrt(px**2 + py**2)")
dsl.alias("eta", "-log(tan(theta/2))")
dsl.alias("phi", "atan2(py, px)")
dsl.alias("high_pt", "pt > 10")
dsl.alias("central", "abs(eta) < 1")
dsl.alias("invariant_mass", "sqrt(E**2 - pt**2)")
# ... many more ...

# Use only what's needed
dsl.draw("high_pt", rdf)  # Only pt, high_pt compiled
dsl.draw("central", rdf)  # Only eta, central compiled

# Example 4: Mixed workflow
dsl = DSLCompiler({'px': 'double', 'py': 'double'})
dsl.define("pt", "sqrt(px**2 + py**2)")  # ✅ Validated (px, py in schema)
dsl.alias("high_pt", "pt > 10")          # Deferred ('pt' added by define())
dsl.alias("track_sum", "Sum(track_pt)")  # Deferred ('track_pt' from RDF)
rdf = dsl.apply(rdf)  # Aliases validated against merged schema
```

### 7.9 Edge Cases and Rules

```python
# ✅ ALLOWED: alias() references define()
dsl.define("pt", "sqrt(px**2 + py**2)")
dsl.alias("high_pt", "pt > 10")  # 'pt' in schema from define()

# ✅ ALLOWED: alias() references alias()
dsl.alias("pt", "sqrt(px**2 + py**2)")
dsl.alias("high_pt", "pt > 10")  # Resolved via dependency tracing

# ✅ ALLOWED: alias() references future RDF column
dsl.alias("track_sum", "Sum(track_pt)")  # 'track_pt' comes from RDF

# ❌ ERROR: define() references alias()
dsl.alias("a", "x + 1")
dsl.define("b", "a * 2")  # IRError: 'a' not in schema

# ❌ ERROR: Circular dependency (detected at materialization)
dsl.alias("a", "b + 1")
dsl.alias("b", "a + 1")
dsl.draw("a", rdf)  # IRError(CYCLE_ERROR)

# ❌ ERROR: Missing column (detected at materialization)
dsl.alias("bad", "nonexistent + 1")
dsl.draw("bad", rdf)  # IRError: 'nonexistent' not in schema or RDF
```

### 7.10 Implementation Outline

```python
class DSLCompiler:
    def __init__(self, schema: Dict[str, str] = None, ...):
        self.schema = schema or {}
        self._definitions = []      # [(name, expr), ...] - validated
        self._aliases = {}          # {name: expr} - deferred (POOL)
        self._compiled = {}         # {name: GeneratedFunction}
        ...
    
    def alias(self, name: str, expression: str) -> 'DSLCompiler':
        """Store alias in pool for deferred validation."""
        if name in self.schema:
            raise IRError(
                IRErrorKind.VALIDATION_ERROR,
                f"Alias '{name}' conflicts with existing column"
            )
        if name in self._aliases:
            raise IRError(
                IRErrorKind.VALIDATION_ERROR,
                f"Alias '{name}' already defined"
            )
        self._aliases[name] = expression
        return self
    
    def _get_needed_aliases(self, requested: List[str]) -> Set[str]:
        """Trace dependencies from requested columns."""
        needed = set()
        visited = set()
        
        def trace(name):
            if name in visited:
                raise IRError(IRErrorKind.CYCLE_ERROR,
                    f"Circular dependency detected involving '{name}'")
            if name in self._aliases and name not in needed:
                visited.add(name)
                needed.add(name)
                for dep in self._extract_dependencies(self._aliases[name]):
                    trace(dep)
                visited.remove(name)
        
        for name in requested:
            trace(name)
        return needed
    
    def _materialize_aliases(self, requested: List[str], rdf) -> None:
        """Validate and compile only needed aliases."""
        # Step 1: Update schema from RDF
        self._update_schema_from_rdf(rdf)
        
        # Step 2: Get needed aliases
        needed = self._get_needed_aliases(requested)
        
        # Step 3: Compile in dependency order
        compiled = set()
        def compile_with_deps(name):
            if name in compiled:
                return
            # Compile dependencies first
            for dep in self._extract_dependencies(self._aliases[name]):
                if dep in needed:
                    compile_with_deps(dep)
            # Now compile this alias
            self._define_with_validation(name, self._aliases[name])
            compiled.add(name)
        
        for name in needed:
            compile_with_deps(name)
    
    def _update_schema_from_rdf(self, rdf) -> None:
        """Add RDF columns to schema (don't overwrite existing)."""
        for col in rdf.GetColumnNames():
            col_name = str(col)
            if col_name not in self.schema:
                self.schema[col_name] = str(rdf.GetColumnType(col_name))
        self._rebuild_inferrer()
```

### 7.11 Benefits

| Benefit | Description |
|---------|-------------|
| **Backward compatible** | `define()` unchanged |
| **Name compatible** | `alias` matches ROOT's `SetAlias` |
| **Template libraries** | Define many, use subset |
| **Lazy validation** | Only validates/compiles what's needed |
| **Clear semantics** | Immediate (`define`) vs deferred (`alias`) |
| **Pool-based** | Unused aliases never cause errors |

---

## 8. For Reviewers: Validation Questions - ANSWERED

### Q1: Schema propagation accurate?
✅ **YES** - Verified against source code (see Section 2)

### Q2: `alias()` semantics clear?
✅ **YES** - See Section 7.3-7.4

### Q3: Validation timing?
**ANSWER:** At first use (`apply()`/`draw()`/`to_pandas()`) - pool-based, only needed aliases

### Q4: Error handling?
**ANSWER:** Collect errors for requested aliases only. Unused aliases never validated.

### Q5: Backward compatibility concerns?
✅ **NONE** - All changes are additive. `define()` unchanged.

### Q6: Naming?
✅ **`alias()` is correct** - Matches ROOT's `SetAlias` terminology

---

## 9. Recommended Changes for Phase 13.6.F

### 9.1 Layer 1: Validation Enhancements

| # | Change | File | Effort | Priority |
|---|--------|------|--------|----------|
| 1 | Add `DSLError = IRError` alias | `__init__.py` | 1 line | P0 |
| 2 | Upgrade suggestions to difflib | `type_inferrer.py` | 10 lines | P0 |
| 3 | Add "Available columns" to error | `ir_errors.py` | 5 lines | P0 |
| 4 | Add validation at `draw()`/`batch_draw()` | `dsl_compiler.py` | 20 lines | P0 |

### 9.2 Schema Flexibility + alias()

| # | Change | File | Effort | Priority |
|---|--------|------|--------|----------|
| 5 | Allow empty schema `DSLCompiler()` | `dsl_compiler.py` | 5 lines | P0 |
| 6 | Add `alias()` method | `dsl_compiler.py` | 15 lines | P0 |
| 7 | Add `_update_schema_from_rdf()` | `dsl_compiler.py` | 20 lines | P0 |
| 8 | Add `_get_needed_aliases()` | `dsl_compiler.py` | 25 lines | P0 |
| 9 | Add `_materialize_aliases()` | `dsl_compiler.py` | 30 lines | P0 |
| 10 | Add `from_rdf()` class method | `dsl_compiler.py` | 20 lines | P1 |

### 9.3 Total Effort Estimate

| Category | Lines | Days |
|----------|-------|------|
| Layer 1 validation | ~36 | 0.5 |
| Schema flexibility + alias() | ~115 | 1.5 |
| Testing | ~200 | 1.0 |
| Documentation | ~50 | 0.5 |
| **Total** | **~400** | **3.5 days** |

---

## 10. Summary: What Exists vs What's Proposed

### 10.1 Current State (v13.6.E)

```
DSLCompiler(schema)     # Schema REQUIRED
    │
    ├── define(name, expr)    # Validates IMMEDIATELY
    │       │
    │       └── IRBuilder.build() → Schema check → IRError if missing
    │
    └── apply(rdf)            # Compiles, applies to RDF
```

### 10.2 Proposed State (v13.6.F)

```
DSLCompiler(schema=None)    # Schema OPTIONAL
    │
    ├── define(name, expr)    # Validates IMMEDIATELY (unchanged)
    │       │
    │       └── IRBuilder.build() → Schema check → IRError if missing
    │
    ├── alias(name, expr)     # Stores in POOL (NEW)
    │       │
    │       └── Store expression only, no validation
    │
    └── apply(rdf) / draw() / to_pandas()   # POOL-BASED materialization
            │
            ├── Determine requested columns
            ├── _get_needed_aliases() ← dependency tracing
            ├── _update_schema_from_rdf(rdf)
            ├── _materialize_aliases()  # Validate + compile ONLY needed
            └── Apply to RDF
```

### 10.3 Compatibility Matrix

| Workflow | v13.6.E | v13.6.F |
|----------|---------|---------|
| `DSLCompiler(schema)` + `define()` | ✅ | ✅ (unchanged) |
| `DSLCompiler({})` + `define()` | ❌ Error | ❌ Error (still need schema for define) |
| `DSLCompiler()` + `alias()` | N/A | ✅ NEW |
| `DSLCompiler(schema)` + `alias()` | N/A | ✅ NEW |
| Mixed `define()` + `alias()` | N/A | ✅ NEW |
| Template library (many aliases, use few) | N/A | ✅ NEW |

---

## Appendix A: Key File Locations

| Purpose | File | Key Lines |
|---------|------|-----------|
| DSLCompiler | `dsl_compiler.py` | 335-548 (init, define) |
| TypeInferrer | `type_inferrer.py` | 366-416 (from_tree, from_schema) |
| Error handling | `ir_errors.py` | 355-392 (unknown_variable_error) |
| Similar names | `type_inferrer.py` | 836-855 (_find_similar_names) |
| IR building | `ir_builder.py` | 675-696 (_visit_Name - validation) |
| Schema update | `dsl_compiler.py` | 632-658 (_register_alias_type) |
| to_pandas | `dsl_compiler.py` | 1993-2097 (calls apply then AsNumpy) |

---

## Appendix B: Test Coverage

| Test Category | File | Count |
|---------------|------|-------|
| IR Builder validation | test_ir_builder.py | ~50 |
| Type inference | test_type_inference.py | ~30 |
| DSL API | test_dsl_api.py | ~40 |
| Invariance (E2E) | test_invariance_*.py | ~200 |

**New tests needed for alias() - see Phase 13.6.F proposal for details**

---

## Appendix C: ROOT Compatibility Reference

| ROOT API | RDataFrameDSL Equivalent | Validation |
|----------|-------------------------|------------|
| `tree->SetAlias(name, expr)` | `dsl.alias(name, expr)` | Deferred (pool-based) |
| `tree->Draw(expr)` | `dsl.draw(expr, rdf)` | At call (triggers materialization) |
| `rdf.Define(name, expr)` | `dsl.define(name, expr)` | Immediate |
| `rdf.GetColumnNames()` | `dsl._update_schema_from_rdf()` | N/A |

---

**Document Status:** APPROVED with v1.2 corrections  
**Version:** 1.2  
**Date:** 2026-01-24  
**Next Step:** Phase 13.6.F Proposal v1.3 (implementation scope, Q5-Q7 acceptance criteria)
