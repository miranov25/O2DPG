# draw() Cheatsheet

Quick reference for `AliasDataFrame.draw()` and related plotting methods.

---

## Return Value

```python
fig, ax, stats = adf.draw('x')

# stats dict keys:
#   n     - entry count
#   mean  - mean value
#   std   - standard deviation
#   min   - minimum value
#   max   - maximum value
```

---

## Plot Types

### 1D Histogram

```python
# Basic histogram
adf.draw('x')

# Custom bins
adf.draw('x', bins=100)

# Custom range
adf.draw('x', bins=100, range=(-10, 10))

# Explicit method
adf.hist('x', bins=50)
```

### 2D Scatter

```python
# Syntax: 'y:x' (y vs x)
adf.draw('y:x')

# Explicit method
adf.scatter('y:x')
```

### 2D Histogram

```python
adf.draw('y:x', type='hist2d', bins=100)

# Different bins per axis
adf.hist2d('y:x', bins=[50, 100])
```

### Profile (mean y vs x)

```python
adf.draw('y:x', type='profile', bins=50)

# Explicit method
adf.profile('y:x', bins=50)
```

### Hexbin

```python
adf.hexbin('y:x', gridsize=50)
```

---

## Selections

```python
# Simple cut
adf.draw('x', selection='pt > 0.5')

# Multiple conditions
adf.draw('x', selection='isOK && charge > 0')

# Complex expression
adf.draw('x', selection='(pt > 0.5) & (eta < 1.0)')
```

---

## Entry Range (Quick Testing)

```python
# First 10k entries only
adf.draw('x', entry_begin=0, entry_end=10000)

# Skip first 1000
adf.draw('x', entry_begin=1000)
```

---

## Group By (Overlaid Histograms)

```python
# Overlay by category
adf.draw('x', group_by='charge')

# With custom bins
adf.draw('x', group_by='sector', bins=50)
```

---

## Color Mapping

```python
# Color points by third variable
adf.draw('y:x', color='z')

# Scatter with color
adf.scatter('y:x', color='pt')
```

---

## With Aliases

```python
# Define alias
adf.add_alias('pt', 'np.sqrt(px**2 + py**2)')

# Option 1: Draw expression directly (no materialization needed)
adf.draw('np.sqrt(px**2 + py**2)')

# Option 2: Materialize alias first, then draw
adf.materialize_alias('pt')  # Creates column in df
adf.draw('pt')  # Now works!

# Alias in expression
adf.draw('pt:eta')  # Works if pt is materialized
```

**Important**: `adf.draw('alias_name')` requires the alias to be materialized first!

---

## With Subframes

```python
# Register calibration table
adf.register_subframe('Calib', calib_adf, index_columns='sec')

# Define alias using subframe
adf.add_alias('corrected', 'signal * Calib.gain')

# Draw with subframe join (works automatically!)
adf.draw('corrected:sec', type='profile')
```

### Lazy Subframes

```python
# Register without loading
adf.register_subframe_lazy('Calib', 'calib.root', 'tree', index_columns='sec')

# Draw triggers automatic loading
adf.draw('corrected')  # Loads subframe on demand
```

---

## Batch Drawing

```python
# Multiple plots at once
specs = {
    'plot1': {'expr': 'x', 'bins': 50},
    'plot2': {'expr': 'y:x', 'type': 'scatter'},
    'plot3': {'expr': 'pt', 'selection': 'charge > 0'},
}
results = adf.draw_batch(specs)

# Access results
fig1, ax1, stats1 = results['plot1']
```

---

## Lazy Mode Integration

```python
# Lazy loading - branches loaded on demand
adf = AliasDataFrame.read_tree_lazy('data.root', 'tree')

# draw() auto-loads required branches
adf.draw('x')  # Loads 'x' automatically

# Check what's loaded
print(adf.loaded_branches)
```

---

## Common Patterns

### QA Plot

```python
fig, ax, stats = adf.draw('residual:sector', type='profile', bins=36)
ax.axhline(0, color='red', linestyle='--')
ax.set_title(f'Residuals (n={stats["n"]:,})')
```

### Before/After Comparison

```python
adf.add_alias('raw', 'signal')
adf.add_alias('corrected', 'signal * Calib.gain')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
adf.draw('raw:sec', type='profile', ax=axes[0])
adf.draw('corrected:sec', type='profile', ax=axes[1])
```

### Save Figure

```python
fig, ax, stats = adf.draw('x')
fig.savefig('histogram.png', dpi=150)
```

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: Branches not found` | Lazy mode, branch doesn't exist | Check branch name with `adf.available_branches` |
| `ValueError: Alias cycle detected` | Circular alias dependency | Check alias definitions |
| `KeyError` in subframe | Missing calibration key | Returns NaN (by design) |
| `KeyError: 'mean'` in stats | Profile/2D plots have different stats | Use `stats.get('mean')` or compute from df |
| `ModuleNotFoundError: dfdraw` | dfdraw not installed | `pip install dfdraw` |
| `UndefinedVariableError` | Drawing unmaterialized alias | Call `adf.materialize_alias()` first |

---

## Stats Dict Reference

```python
fig, ax, stats = adf.draw('x')

# For 1D histograms - all keys available:
stats['n']      # int: number of entries
stats['mean']   # float: mean value
stats['std']    # float: standard deviation
stats['min']    # float: minimum value
stats['max']    # float: maximum value

# For 2D plots (scatter, profile, hist2d):
# Stats dict may have different/fewer keys
# Always use stats.get('key', default) for safety
```

**Note**: Profile and 2D plots return different stats than 1D histograms.
Use `stats.get('mean', None)` or check `'mean' in stats` to avoid KeyError.

---

## See Also

- `tutorials/drawing/` — Full examples
- `tests/test_draw_invariance.py` — Test cases
- `tests/test_draw_chain_integration.py` — Chain + subframe tests
