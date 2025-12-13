#!/usr/bin/env python3
"""
AliasDataFrame Tutorial: Selections and Group By

Topics: selection parameter, entry range, group_by overlays
Data: Inline NumPy (no file dependencies)
Phase: 6.8 (dfdraw integration)

Run:
    python selections_groupby.py

Expected output:
    Statistics for filtered and grouped plots
    Demonstrates selection strings and overlays
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directories for imports
# AliasDataFrame is 2 levels up, dfdraw is 3 levels up (sibling of AliasDataFrame)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TUTORIALS_DIR = os.path.dirname(SCRIPT_DIR)
ADF_DIR = os.path.dirname(TUTORIALS_DIR)
DFEXTENSIONS_DIR = os.path.dirname(ADF_DIR)

sys.path.insert(0, ADF_DIR)          # For AliasDataFrame
sys.path.insert(0, DFEXTENSIONS_DIR) # For dfdraw

from AliasDataFrame import AliasDataFrame

# Check if dfdraw is available
try:
    from dfdraw import DFDraw
    HAS_DFDRAW = True
except ImportError:
    HAS_DFDRAW = False
    print("Note: dfdraw not installed. Stats will be shown, plots skipped.")

# =============================================================================
# Setup: Create sample data with categories
# =============================================================================

print("=" * 60)
print("Setting up sample data with categories...")
print("=" * 60)

np.random.seed(42)
n_rows = 20000

# Simulate particle physics data
charge = np.random.choice([-1, 0, 1], n_rows, p=[0.3, 0.4, 0.3])
pt = np.abs(np.random.exponential(1.5, n_rows))  # Transverse momentum
eta = np.random.uniform(-2, 2, n_rows)           # Pseudorapidity
sector = np.random.randint(0, 36, n_rows)        # Detector sector
is_good = np.random.choice([True, False], n_rows, p=[0.8, 0.2])

df = pd.DataFrame({
    'charge': charge,
    'pt': pt,
    'eta': eta,
    'sector': sector,
    'isGood': is_good.astype(int),  # Boolean as int for selection
    'quality': np.random.randint(0, 4, n_rows)   # Quality flag 0-3
})

adf = AliasDataFrame(df)
print(f"Created AliasDataFrame with {len(adf.df):,} rows")
print(f"Charge distribution: {dict(zip(*np.unique(charge, return_counts=True)))}")
print()

# =============================================================================
# Example 1: Basic selection
# =============================================================================

print("=" * 60)
print("Example 1: Basic selection")
print("=" * 60)

if HAS_DFDRAW:
    # Select only positive charge particles
    fig, ax, stats = adf.draw('pt', selection='charge > 0')
    print(f"adf.draw('pt', selection='charge > 0')")
    print(f"Entries with charge > 0: {stats['n']:,}")
else:
    n_positive = (df['charge'] > 0).sum()
    print(f"Entries with charge > 0: {n_positive:,}")
print()

# =============================================================================
# Example 2: Multiple conditions
# =============================================================================

print("=" * 60)
print("Example 2: Multiple conditions")
print("=" * 60)

if HAS_DFDRAW:
    # Combine conditions with & (AND) or | (OR) - Python syntax
    fig, ax, stats = adf.draw('pt', selection='(charge != 0) & (pt > 0.5)')
    print(f"selection='(charge != 0) & (pt > 0.5)'")
    print(f"Charged particles with pt > 0.5: {stats['n']:,}")
print()

# =============================================================================
# Example 3: Using Python syntax
# =============================================================================

print("=" * 60)
print("Example 3: Python-style selection")
print("=" * 60)

if HAS_DFDRAW:
    # Use & for AND, | for OR (Python/pandas syntax)
    fig, ax, stats = adf.draw('eta', selection='(pt > 1.0) & (isGood == 1)')
    print(f"selection='(pt > 1.0) & (isGood == 1)'")
    print(f"Good particles with pt > 1: {stats['n']:,}")
print()

# =============================================================================
# Example 4: Entry range (quick testing)
# =============================================================================

print("=" * 60)
print("Example 4: Entry range for quick testing")
print("=" * 60)

if HAS_DFDRAW:
    # Process only first 1000 entries (fast iteration)
    fig, ax, stats = adf.draw('pt', entry_begin=0, entry_end=1000)
    print(f"adf.draw('pt', entry_begin=0, entry_end=1000)")
    print(f"Processed only first 1000 entries: {stats['n']:,}")
print()

# =============================================================================
# Example 5: Entry range + selection
# =============================================================================

print("=" * 60)
print("Example 5: Combining entry range and selection")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('pt', 
                              entry_begin=0, 
                              entry_end=5000,
                              selection='charge > 0')
    print(f"First 5000 entries, charge > 0")
    print(f"Entries after both filters: {stats['n']:,}")
print()

# =============================================================================
# Example 6: Group by (overlaid histograms)
# =============================================================================

print("=" * 60)
print("Example 6: Group by - overlaid histograms")
print("=" * 60)

if HAS_DFDRAW:
    # Overlay histograms for each charge value
    fig, ax, stats = adf.draw('pt', group_by='charge', bins=50)
    print(f"adf.draw('pt', group_by='charge')")
    print(f"Creates separate histogram for each charge value")
    print(f"Total entries: {stats['n']:,}")
print()

# =============================================================================
# Example 7: Group by with selection
# =============================================================================

print("=" * 60)
print("Example 7: Group by + selection")
print("=" * 60)

if HAS_DFDRAW:
    # Group by quality, but only for charged particles
    fig, ax, stats = adf.draw('pt', 
                              group_by='quality',
                              selection='charge != 0',
                              bins=50)
    print(f"group_by='quality', selection='charge != 0'")
    print(f"Overlaid by quality flag, charged only")
print()

# =============================================================================
# Example 8: 2D plot with selection
# =============================================================================

print("=" * 60)
print("Example 8: 2D plot with selection")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('pt:eta', 
                              selection='isGood == 1',
                              type='hist2d',
                              bins=50)
    print(f"adf.draw('pt:eta', selection='isGood == 1', type='hist2d')")
    print(f"2D histogram of good particles only")
print()

# =============================================================================
# Example 9: Profile with group_by
# =============================================================================

print("=" * 60)
print("Example 9: Profile with group_by")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('pt:sector', 
                              type='profile',
                              group_by='charge',
                              bins=36)
    print(f"type='profile', group_by='charge'")
    print(f"Mean pt vs sector, separate line per charge")
print()

# =============================================================================
# Example 10: Selection with aliases
# =============================================================================

print("=" * 60)
print("Example 10: Selection using aliases")
print("=" * 60)

# Define alias
adf.add_alias('pt_cut', 'pt > 1.0')

if HAS_DFDRAW:
    # Can use alias in selection (if materialized)
    fig, ax, stats = adf.draw('eta', selection='pt > 1.0')
    print(f"Selection: pt > 1.0")
    print(f"High-pt particles: {stats['n']:,}")
print()

# =============================================================================
# Example 11: Boolean mask selection
# =============================================================================

print("=" * 60)
print("Example 11: Entry mask selection")
print("=" * 60)

# Create a custom mask
custom_mask = (df['pt'] > 1.0) & (df['charge'] != 0)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('eta', entry_mask=custom_mask.values)
    print(f"Using entry_mask parameter with numpy boolean array")
    print(f"Entries matching mask: {stats['n']:,}")
else:
    print(f"Entries matching mask: {custom_mask.sum():,}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 60)
print("Summary")
print("=" * 60)
print("""
Selection syntax:
  selection='x > 0'              Single condition
  selection='(x > 0) & (y < 1)'  AND (use & with parentheses)
  selection='(x > 0) | (y > 0)'  OR (use | with parentheses)

  Note: Use Python syntax (&, |), NOT C++ syntax (&&, ||)

Entry range (for fast iteration):
  entry_begin=0, entry_end=1000

Group by (overlaid plots):
  group_by='column'           Separate plot per unique value

Entry mask (programmatic):
  entry_mask=boolean_array    Direct numpy mask

Combining:
  All parameters can be combined!
  selection + group_by + entry_range all work together

Next: See with_subframes.py for calibration joins
""")

# Show plots if running interactively
if HAS_DFDRAW:
    try:
        import matplotlib.pyplot as plt
        # Uncomment to show plots:
        # plt.show()
        print("(Plots created but not displayed. Uncomment plt.show() to view)")
    except:
        pass
