# AliasDataFrame Tutorials

Quick start guide for AliasDataFrame + dfdraw.

---

## Prerequisites

- **AliasDataFrame** installed
- **dfdraw** installed (for plotting)
- For subframe examples: ROOT files from `data/`

### Quick Install Check

```bash
cd ~/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame

# Set PYTHONPATH (required!)
export PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH"

# Verify imports
python3 -c "from AliasDataFrame import AliasDataFrame; from dfdraw import DFDraw; print('Ready!')"
```

---

## Running Tutorials

**Always run from the AliasDataFrame directory with PYTHONPATH set:**

```bash
cd ~/alicesw/O2DPG/UTILS/dfextensions/AliasDataFrame

# Option 1: Run all tutorials automatically
PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH" ./tutorials/test_tutorials.sh

# Option 2: Run individual tutorial
PYTHONPATH="$PWD:$PWD/..:$PYTHONPATH" python3 tutorials/drawing/histograms.py
```

---

## Where to Start

| Time | Goal | Location |
|------|------|----------|
| 5 min | Quick syntax reference | `cheatsheets/draw_cheatsheet.md` |
| 15 min | Learn drawing | `drawing/` |
| 30 min | Real workflow | `workflows/` |

---

## Folder Structure

```
tutorials/
├── README.md              ← You are here
├── cheatsheets/
│   └── draw_cheatsheet.md ← Quick reference (copy-paste ready)
├── drawing/
│   ├── histograms.py      ← 1D histograms, binning, stats
│   ├── scatter_profile.py ← 2D plots, profiles, hexbin
│   ├── selections_groupby.py ← Filtering, overlays
│   └── with_subframes.py  ← Calibration joins + draw
├── quickstart/
│   └── minimal_example.py ← Complete workflow in 50 lines
├── workflows/
│   └── tpc_calibration.py ← Real ALICE calibration example
└── data/
    └── README.md          ← How to generate sample data
```

---

## Learning Path

### Beginner (30 minutes)

1. Read `cheatsheets/draw_cheatsheet.md`
2. Run `drawing/histograms.py`
3. Run `drawing/scatter_profile.py`

### Intermediate (1 hour)

4. Run `drawing/selections_groupby.py`
5. Generate sample data (see `data/README.md`)
6. Run `drawing/with_subframes.py`

### Advanced (2+ hours)

7. Study `workflows/tpc_calibration.py`
8. Adapt to your own data

---

## About dfdraw

AliasDataFrame uses **dfdraw** as its plotting backend. The `adf.draw()` method wraps dfdraw functionality.

**You do NOT need to know dfdraw internals to use `adf.draw()`.**

The cheatsheet documents the supported API surface:
- All plot types: hist, scatter, profile, hist2d, hexbin
- Selections, group_by, color mapping
- Stats dict return values

---

## Key Concepts

### 1. Draw Syntax

```python
# 1D: just column name
adf.draw('x')

# 2D: 'y:x' syntax (y vs x)
adf.draw('y:x')
```

### 2. Return Values

```python
fig, ax, stats = adf.draw('x')
# fig: matplotlib Figure
# ax: matplotlib Axes
# stats: dict with n, mean, std, min, max
```

### 3. Lazy Loading Integration

```python
# Branches load automatically when needed
adf = AliasDataFrame.read_tree_lazy('data.root', 'tree')
adf.draw('x')  # Loads 'x' on demand
```

### 4. Subframe Joins

```python
# Calibration data joins automatically
adf.add_alias('corrected', 'signal * Calib.gain')
adf.draw('corrected')  # Joins, computes, draws
```

---

## Getting Help

- **API Reference**: Check docstrings with `help(adf.draw)`
- **Test Cases**: See `tests/test_draw_*.py` for working examples
- **Phase History**: See `PHASE_HISTORY.md` for architecture context

---

## Contributing Examples

When creating new tutorials:

1. Use inline data for simple examples (no file dependencies)
2. Use `generate_synthetic_data.py` for complex examples
3. Include expected output in comments
4. Follow the header template in existing files
