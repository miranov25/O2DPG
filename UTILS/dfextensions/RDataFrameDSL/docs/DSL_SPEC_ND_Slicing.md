# DSL Specification: N-D Slicing with Join Strategy

**Document:** DSL_SPEC_ND_Slicing.md  
**Version:** 1.0  
**Status:** Official Specification  
**Effective:** Phase 13.6.C  
**Last Updated:** 2026-01-18

---

## 1. Overview

This document specifies the N-dimensional slicing and join semantics for RDataFrameDSL. It defines how nested particle physics data structures (events → tracks → clusters → hits → ...) can be sliced, combined, and exported to pandas DataFrames.

### 1.1 Design Principles

1. **NumPy-compatible syntax** — Familiar slicing notation
2. **Uniform N-D pattern** — Same rules for 2D, 3D, and beyond
3. **Inner join default** — Conservative, no unexpected NaN
4. **Per-parent evaluation** — Slices apply within parent groups
5. **Deterministic output** — Reproducible results

---

## 2. Data Hierarchy Model

### 2.1 Hierarchy Structure

```
event (0D) → track (1D) → cluster (2D) → hit (3D) → ...
     │            │             │            │
  scalar    [track_idx]  [track_idx,    [track_idx,
                          cluster_idx]   cluster_idx,
                                         hit_idx]
```

Each hierarchy level has a **depth**:

| Level | Depth | Index Columns |
|-------|-------|---------------|
| event | 0 | `event_id` |
| track | 1 | `event_id`, `track_idx` |
| cluster | 2 | `event_id`, `track_idx`, `cluster_idx` |
| hit | 3 | `event_id`, `track_idx`, `cluster_idx`, `hit_idx` |

### 2.2 Maximum Depth

Supported dimension depth: **Up to 10D** (practical limit for physics data).

---

## 3. Slicing Syntax

### 3.1 Basic Syntax

Slicing follows NumPy conventions:

```
column[start:stop:step]           # 1D
column[start:stop, start:stop]    # 2D
column[i, j, k]                   # 3D with single indices
column[i:j, :, k:l]               # 3D with mixed slices
```

### 3.2 Supported Slice Types

| Type | Syntax | Example | Meaning |
|------|--------|---------|---------|
| Single index | `[i]` | `track[0]` | Element at index i |
| Negative index | `[-i]` | `track[-1]` | i-th from end |
| Range | `[start:stop]` | `track[0:3]` | Elements 0, 1, 2 |
| Open start | `[:stop]` | `track[:3]` | Elements 0, 1, 2 |
| Open end | `[start:]` | `track[2:]` | From 2 to end |
| Full range | `[:]` | `track[:]` | All elements |
| Step | `[::step]` | `track[::2]` | Every other element |
| Negative step | `[::-1]` | `track[::-1]` | Reversed |
| Negative range | `[-3:]` | `track[-3:]` | Last 3 elements |

### 3.3 Multi-Dimensional Syntax

**2D (track → cluster):**
```python
cluster[0, :]       # All clusters from track 0
cluster[:, 0]       # First cluster from each track
cluster[0:2, :]     # All clusters from tracks 0, 1
cluster[:, 0:2]     # First 2 clusters from each track
cluster[0:2, 0:3]   # First 3 clusters from tracks 0, 1
cluster[-1, :]      # All clusters from last track
cluster[:, -1]      # Last cluster from each track
cluster[::2, :]     # Clusters from every other track
```

**3D (track → cluster → hit):**
```python
hit[0, :, :]        # All hits from track 0
hit[:, 0, :]        # All hits from first cluster of each track
hit[:, :, 0]        # First hit from each cluster
hit[0:2, :, 0:3]    # First 3 hits from all clusters of tracks 0, 1
hit[-1, -1, -1]     # Last hit of last cluster of last track
```

### 3.4 Shortcuts

Omitted trailing dimensions default to `[:]`:

| Shortcut | Equivalent | Meaning |
|----------|------------|---------|
| `track` | `track[:]` | All tracks |
| `cluster` | `cluster[:, :]` | All clusters |
| `cluster[0]` | `cluster[0, :]` | Track 0, all clusters |
| `hit` | `hit[:, :, :]` | All hits |
| `hit[0]` | `hit[0, :, :]` | Track 0, all clusters, all hits |
| `hit[0, 1]` | `hit[0, 1, :]` | Track 0, cluster 1, all hits |

### 3.5 Feature Matrix

| Feature | Syntax Example | Supported |
|---------|----------------|-----------|
| Positive index | `cluster[0, 1]` | ✅ |
| Negative index | `cluster[-1, -2]` | ✅ |
| Range slice | `cluster[0:2, 1:4]` | ✅ |
| Step slice | `cluster[::2, ::3]` | ✅ |
| Open start | `cluster[:2, :3]` | ✅ |
| Open end | `cluster[1:, 2:]` | ✅ |
| Full range | `cluster[:, :]` | ✅ |
| Negative range | `cluster[-3:, -2:]` | ✅ |
| Reverse | `cluster[::-1, ::-1]` | ✅ |
| Mixed | `cluster[0:2, -1]` | ✅ |
| 3D+ slicing | `hit[0:2, :, 0:3]` | ✅ |
| Boolean mask | `cluster[mask, :]` | ⏳ Future |

---

## 4. Slice Evaluation Scope

### 4.1 Fundamental Rule

**All slicing is evaluated WITHIN PARENT GROUPS.**

Slices are applied independently within each parent context, never across boundaries.

### 4.2 Scoping Hierarchy

```
track[i]       → i-th track WITHIN each event
cluster[i, j]  → i-th track WITHIN event, j-th cluster WITHIN that track
hit[i, j, k]   → within event, within track, within cluster
```

### 4.3 Examples

```python
track[-1]        # Last track PER EVENT (not global last)
cluster[:, -1]   # Last cluster PER (event_id, track_idx)
hit[:, :, 0:2]   # First 2 hits PER (event_id, track_idx, cluster_idx)
```

### 4.4 Invariants

1. Indices **never** cross event boundaries
2. Indices **never** cross parent boundaries
3. Empty parents produce no output (not an error)

---

## 5. Join Strategy

### 5.1 Join Types

| Join Type | Behavior | Default |
|-----------|----------|---------|
| `'inner'` | Intersection of indices | **Yes** |
| `'outer'` | Union of indices, NaN for missing | No |
| `'left'` | All from first/deeper operand | No |
| `'right'` | All from second/shallower operand | No |

### 5.2 Join Rules

**Rule 0 (Foundational): Join Domain Invariant**

All joins implicitly include `event_id`:
- Join key = `(event_id, common_index_columns...)`
- No cross-event joins permitted
- Each event is processed independently

**Rule 1: Scalars broadcast to all levels**
```python
event.weight * track.pt * cluster.Q * hit.E   # ✅ OK
```

**Rule 2: Same-level arrays — element-wise or join**
```python
track[0:2].pt + track[0:2].eta   # Direct element-wise (same slice)
track[0:3].pt + track[0:5].eta   # Inner join → indices 0, 1, 2
```

**Rule 3: Parent-child join on common indices**
```python
cluster[0:3, :].Q / track[0:2].Qexp
# Join on track_idx: {0,1,2} ∩ {0,1} = {0,1}
```

**Rule 4: Full range (`:`) behavior**

Full range is "accepting" — takes whatever the join provides:

| Join Type | `cluster[:, :].Q / track[0:2].Qexp` |
|-----------|-------------------------------------|
| `'inner'` | Clusters from tracks 0, 1 |
| `'outer'` | All clusters; NaN for Qexp where track ∉ {0,1} |
| `'left'` | All clusters included |
| `'right'` | Only clusters from tracks 0, 1 |

### 5.3 Join Computation Scope

Join computation is **global (N-way)** across all operands:

```python
track[0:2].pt + track[1:3].eta + track[2:4].phi

# Index sets: pt={0,1}, eta={1,2}, phi={2,3}
# inner: {0,1} ∩ {1,2} ∩ {2,3} = ∅
# outer: {0,1} ∪ {1,2} ∪ {2,3} = {0,1,2,3}
```

Join is **not** computed pairwise — all operands are joined simultaneously.

---

## 6. NaN Policy

### 6.1 Scope

Applies to `join='outer'`, `join='left'`, `join='right'` only.

`join='inner'` never produces NaN from join logic.

### 6.2 Representation

**Data Columns:**
- Missing values → `NaN` (IEEE 754 float)
- Integer data may promote to `float64` if NaN introduced

**Index Columns:**
- **Never** contain NaN
- Always non-null integers

### 6.3 Example

```python
# track[0:2].pt outer-joined with track[1:3].eta

| event_id | track_idx | pt   | eta  |
|----------|-----------|------|------|
| 0        | 0         | 1.5  | NaN  |  ← track 0 missing eta
| 0        | 1         | 2.5  | 0.3  |  ← both present
| 0        | 2         | NaN  | 0.8  |  ← track 2 missing pt
```

---

## 7. Output Specification

### 7.1 Deepest Level Determination

**Definition:** `hierarchy_depth` = number of index dimensions in the column's schema.

**Rule:** Output level = `max(hierarchy_depth)` across all operands.

| Expression | Deepest Level | Output Indices |
|------------|---------------|----------------|
| `event.weight * 2.0` | event (0) | `event_id` |
| `track.pt * event.weight` | track (1) | `event_id`, `track_idx` |
| `cluster.Q / track.Qexp` | cluster (2) | `event_id`, `track_idx`, `cluster_idx` |
| `hit.E * cluster.Q` | hit (3) | `event_id`, `track_idx`, `cluster_idx`, `hit_idx` |

**Note:** Slicing reduces data, not schema depth. `cluster[:, 0]` is still depth 2.

### 7.2 Row Count

One row per element at the deepest level after join.

### 7.3 Value Replication

Shallower values replicate to match deeper level:

```
event.weight (0D)  →  replicated to every row
track.pt (1D)      →  replicated to every cluster/hit of that track
cluster.Q (2D)     →  replicated to every hit of that cluster
```

### 7.4 Ordering Guarantee

Output rows are sorted by index columns in ascending order:

```
ORDER BY event_id ASC, idx0 ASC, idx1 ASC, ...
```

**Properties:**
- Deterministic across runs
- Reproducible for testing
- Negative indices converted to positive in output

---

## 8. API Reference

### 8.1 Parameters

| Parameter | Purpose | Options | Default |
|-----------|---------|---------|---------|
| `format` | Output structure | `'flat'`, `'dict'`, `'aliasdf'` | `'flat'` |
| `join` | Join strategy | `'inner'`, `'outer'`, `'left'`, `'right'` | `'inner'` |

### 8.2 Functions

**`draw(rdf, expression, join='inner')`**

Evaluate single expression, return DataFrame.

```python
df = dsl.draw(rdf, "cluster.Q / track.Qexp", join='inner')
```

**`batch_draw(rdf, expressions, join='inner')`**

Evaluate multiple expressions, return DataFrame with one column per expression.

```python
df = dsl.batch_draw(rdf, ["cluster.Q", "track.pt", "event.weight"])
```

**`to_pandas(rdf, columns, format='flat', join='inner')`**

Export columns to pandas DataFrame(s).

```python
# Flat DataFrame (default)
df = dsl.to_pandas(rdf, ["track.pt", "cluster.Q"])

# Dict of DataFrames by level
tables = dsl.to_pandas(rdf, ["track.pt", "cluster.Q"], format='dict')

# Lazy AliasDataFrame
adf = dsl.to_pandas(rdf, ["track.pt", "cluster.Q"], format='aliasdf')
```

### 8.3 Output Formats

| Format | Returns | Description |
|--------|---------|-------------|
| `'flat'` | `DataFrame` | Single joined DataFrame at deepest level |
| `'dict'` | `Dict[str, DataFrame]` | Separate DataFrame per hierarchy level |
| `'aliasdf'` | `AliasDataFrame` | Lazy wrapper, joins on access |

---

## 9. Error Handling

### 9.1 Out-of-Bounds Slices

**Behavior:** NumPy-style clipping (silent).

```python
track[0:1000]  # With 100 tracks → track[0:100]
track[500:600] # With 100 tracks → empty result
```

### 9.2 Dimension Mismatch

**Behavior:** `ValueError`

```python
cluster[0:2, 1:3, 2:4]  # 3D slice on 2D data
# ValueError: Slice has 3 dimensions, but 'cluster' has depth 2
```

### 9.3 Empty Join Results

**Behavior:** Return empty DataFrame with correct schema.

```python
cluster[0:2, :].Q / track[5:10].Qexp  # Disjoint indices
# Returns: DataFrame with 0 rows, correct columns and dtypes
```

### 9.4 Invalid Syntax

**Behavior:** `SyntaxError` at parse time.

```python
track[0:2:3:4]  # Too many colons
# SyntaxError: Invalid slice syntax
```

### 9.5 Unknown Column

**Behavior:** `KeyError`

```python
track.nonexistent
# KeyError: Column 'nonexistent' not found in schema
```

---

## 10. Examples

### 10.1 Basic Slicing

```python
# First 5 tracks per event
df = dsl.draw(rdf, "track[0:5].pt")

# Last cluster from each track
df = dsl.draw(rdf, "cluster[:, -1].Q")

# First 3 hits from first 2 clusters of first 2 tracks
df = dsl.draw(rdf, "hit[0:2, 0:2, 0:3].E")
```

### 10.2 Mixed-Depth Expressions

```python
# Cluster charge normalized by track expected charge
df = dsl.draw(rdf, "cluster.Q / track.Qexp * event.weight")

# With explicit slicing
df = dsl.draw(rdf, "cluster[0:5, :].Q / track[0:5].Qexp")
```

### 10.3 Join Strategies

```python
# Inner join (default) — only matching indices
df = dsl.draw(rdf, "track[0:3].pt / track[0:5].eta", join='inner')
# Result: tracks 0, 1, 2

# Outer join — all indices, NaN for missing
df = dsl.draw(rdf, "track[0:3].pt / track[0:5].eta", join='outer')
# Result: tracks 0, 1, 2, 3, 4 (NaN where missing)
```

### 10.4 Multi-Column Export

```python
# Flat DataFrame (default)
df = dsl.to_pandas(rdf, [
    "event.weight",
    "track[0:10].pt",
    "cluster[0:10, :].Q"
])

# Separate tables per level
tables = dsl.to_pandas(rdf, [
    "event.weight",
    "track.pt",
    "cluster.Q"
], format='dict')

# Access by level
event_df = tables['event']
track_df = tables['track']
cluster_df = tables['cluster']
```

---

## 11. Quick Reference

```
SLICING
  track[0]         → First track per event
  track[-1]        → Last track per event
  cluster[0:2, :]  → All clusters from tracks 0,1
  cluster[:, -1]   → Last cluster per track
  hit[0, :, 0:3]   → First 3 hits from all clusters of track 0

JOINS (default: inner)
  inner  → Intersection of indices (no NaN)
  outer  → Union of indices (NaN for missing)
  left   → All from deeper operand
  right  → All from shallower operand

OUTPUT FORMAT (default: flat)
  flat    → Single DataFrame, joined
  dict    → Dict of DataFrames by level
  aliasdf → Lazy AliasDataFrame

API
  draw(rdf, "expr")                    → DataFrame
  batch_draw(rdf, ["expr1", "expr2"])  → DataFrame
  to_pandas(rdf, ["col1", "col2"])     → DataFrame (flat)
  to_pandas(rdf, [...], format='dict') → Dict[str, DataFrame]

RULES
  1. All slicing is per-parent (within groups)
  2. All joins include event_id (no cross-event)
  3. Output at max(depth) of all operands
  4. Rows sorted by (event_id, idx0, idx1, ...)
```

---

## 12. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-18 | Initial specification (Phase 13.6.C) |

---

**END OF SPECIFICATION**
