# Tutorial Testing Guide

## Quick Test (Automated)

```bash
cd /path/to/AliasDataFrame

# Run all tests automatically
python tutorials/test_tutorials.py

# With verbose output
python tutorials/test_tutorials.py -v

# Force regenerate data
python tutorials/test_tutorials.py --force
```

---

## Manual Testing Steps

### Step 0: Prerequisites Check

```bash
# Check AliasDataFrame
python -c "from AliasDataFrame import AliasDataFrame; print('OK')"

# Check uproot
python -c "import uproot; print('OK')"

# Check dfdraw (optional)
python -c "from dfdraw import DFDraw; print('OK')"
```

### Step 1: Generate Data

```bash
cd /path/to/AliasDataFrame

# Generate synthetic data for tutorials
python benchmarks/generate_synthetic_data.py \
    -o tutorials/data/synthetic_data.root \
    --rows 10000 \
    --verify

# Expected output:
# Generating synthetic ROOT file: tutorials/data/synthetic_data.root
#   Seed: 42 (base=42, file_index=0)
#   Main tree rows: 10,000
#   ...
# ✓ Invariant verified: y_derived = 2 * x
```

### Step 2: Test histograms.py

```bash
cd tutorials/drawing
python histograms.py

# Expected output:
# ============================================================
# Setting up sample data...
# ============================================================
# Created AliasDataFrame with 10,000 rows
# ...
# Summary
# ============================================================
# Key points:
# 1. adf.draw('column') creates a histogram
# ...
```

### Step 3: Test scatter_profile.py

```bash
python scatter_profile.py

# Expected output:
# ============================================================
# Setting up sample data with correlations...
# ============================================================
# Created AliasDataFrame with 5,000 rows
# Correlations: y ≈ 2*x, z ≈ x²
# ...
```

### Step 4: Test selections_groupby.py

```bash
python selections_groupby.py

# Expected output:
# ============================================================
# Setting up sample data with categories...
# ============================================================
# Created AliasDataFrame with 20,000 rows
# Charge distribution: {-1: 6033, 0: 7967, 1: 6000}
# ...
```

### Step 5: Test with_subframes.py

```bash
python with_subframes.py

# Expected output (with ROOT file):
# ============================================================
# Loading data...
# ============================================================
# Loaded main tree from: .../synthetic_data.root
# Available branches: ['file_idx', 'row', ...]
# Loaded calibration: 36 sectors
# ...
# ✓ Calibration invariant verified!

# Expected output (without ROOT file - fallback):
# ============================================================
# DATA FILE NOT FOUND
# ============================================================
# ...
# Using fallback inline data
# Main data: 5,000 rows
# Calibration: 36 sectors
```

---

## Verification Checklist

| Step | Test | Expected Result |
|------|------|-----------------|
| 0 | Prerequisites | AliasDataFrame, uproot available |
| 1 | Data generation | `synthetic_data.root` created, invariant verified |
| 2 | histograms.py | Runs without error, shows stats |
| 3 | scatter_profile.py | Runs without error, shows stats |
| 4 | selections_groupby.py | Runs without error, shows stats |
| 5 | with_subframes.py | Runs without error, calibration verified |

---

## Common Issues

### "ModuleNotFoundError: AliasDataFrame"

```bash
# Add to PYTHONPATH
export PYTHONPATH=/path/to/AliasDataFrame:$PYTHONPATH

# Or run from AliasDataFrame directory
cd /path/to/AliasDataFrame
python tutorials/drawing/histograms.py
```

### "FileNotFoundError: synthetic_data.root"

```bash
# Generate data first
python benchmarks/generate_synthetic_data.py -o tutorials/data/synthetic_data.root
```

### "dfdraw not available"

Tutorials will still run and show statistics. Plots will be skipped.

```bash
# Install dfdraw (if available)
pip install dfdraw

# Or ignore - tutorials work without it
```

### Plots don't display

Uncomment `plt.show()` at the end of each tutorial:

```python
# In tutorial script, change:
# plt.show()
# To:
plt.show()
```

---

## Success Criteria

All tutorials should:

1. **Run without errors** (exit code 0)
2. **Print statistics** for each example
3. **Show "Summary" section** at the end
4. **Verify invariants** (where applicable)

If using dfdraw, tutorials should also:

5. **Create matplotlib figures** (even if not displayed)
6. **Return (fig, ax, stats) tuples** from draw calls
