#!/usr/bin/env python3
"""
AliasDataFrame Tutorial: Histograms

Topics: 1D histograms, binning, range, stats dict
Data: Inline NumPy (no file dependencies)
Phase: 6.8 (dfdraw integration)

Run:
    python histograms.py

Expected output:
    Statistics for each histogram variant
    Optional: matplotlib figures if display available
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directories for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AliasDataFrame import AliasDataFrame

# Check if dfdraw is available
try:
    from dfdraw import DFDraw
    HAS_DFDRAW = True
except ImportError:
    HAS_DFDRAW = False
    print("Note: dfdraw not installed. Stats will be shown, plots skipped.")

# =============================================================================
# Setup: Create sample data
# =============================================================================

print("=" * 60)
print("Setting up sample data...")
print("=" * 60)

np.random.seed(42)
n_rows = 10000

df = pd.DataFrame({
    'x': np.random.randn(n_rows),                    # Standard normal
    'y': np.random.randn(n_rows) * 2 + 1,            # Scaled and shifted
    'z': np.random.exponential(2, n_rows),           # Exponential
    'category': np.random.choice(['A', 'B', 'C'], n_rows)
})

adf = AliasDataFrame(df)
print(f"Created AliasDataFrame with {len(adf.df):,} rows")
print(f"Columns: {list(adf.df.columns)}")
print()

# =============================================================================
# Example 1: Basic histogram
# =============================================================================

print("=" * 60)
print("Example 1: Basic histogram")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('x')
    print(f"Entries: {stats['n']:,}")
    print(f"Mean:    {stats['mean']:.4f}")
    print(f"Std:     {stats['std']:.4f}")
    print(f"Min:     {stats['min']:.4f}")
    print(f"Max:     {stats['max']:.4f}")
else:
    # Manual stats without dfdraw
    print(f"Entries: {len(df):,}")
    print(f"Mean:    {df['x'].mean():.4f}")
    print(f"Std:     {df['x'].std():.4f}")
print()

# =============================================================================
# Example 2: Custom binning
# =============================================================================

print("=" * 60)
print("Example 2: Custom binning")
print("=" * 60)

if HAS_DFDRAW:
    # More bins for finer resolution
    fig, ax, stats = adf.draw('x', bins=100)
    print(f"Using 100 bins (default is ~50)")
    print(f"Stats: n={stats['n']:,}, mean={stats['mean']:.3f}")
print()

# =============================================================================
# Example 3: Custom range
# =============================================================================

print("=" * 60)
print("Example 3: Custom range")
print("=" * 60)

if HAS_DFDRAW:
    # Restrict to ±3 sigma
    fig, ax, stats = adf.draw('x', bins=50, range=(-3, 3))
    print(f"Range restricted to (-3, 3)")
    print(f"Entries in range: {stats['n']:,} (may be less than total)")
print()

# =============================================================================
# Example 4: Using .hist() explicitly
# =============================================================================

print("=" * 60)
print("Example 4: Using .hist() method")
print("=" * 60)

if HAS_DFDRAW:
    # Equivalent to draw() for 1D
    fig, ax, stats = adf.hist('y', bins=80)
    print(f"adf.hist('y', bins=80)")
    print(f"Stats: n={stats['n']:,}, mean={stats['mean']:.3f}, std={stats['std']:.3f}")
print()

# =============================================================================
# Example 5: Drawing an expression (computed on-the-fly)
# =============================================================================

print("=" * 60)
print("Example 5: Drawing an expression")
print("=" * 60)

if HAS_DFDRAW:
    # Draw expression directly - dfdraw evaluates it
    fig, ax, stats = adf.draw('x**2', bins=50)
    print(f"Expression: 'x**2'")
    print(f"Stats: n={stats['n']:,}, mean={stats['mean']:.3f}")
    print(f"(Expected mean ≈ 1.0 for standard normal squared)")
else:
    print(f"Expression x**2: mean={df['x'].pow(2).mean():.3f}")
print()

# =============================================================================
# Example 6: Drawing a materialized alias
# =============================================================================

print("=" * 60)
print("Example 6: Drawing a materialized alias")
print("=" * 60)

# Define AND materialize the alias (adds column to df)
adf.add_alias('x_squared', 'x**2')
adf.materialize_alias('x_squared')

if HAS_DFDRAW:
    # Now x_squared exists as a real column
    fig, ax, stats = adf.draw('x_squared', bins=50)
    print(f"Alias 'x_squared' = x**2 (materialized)")
    print(f"Stats: n={stats['n']:,}, mean={stats['mean']:.3f}")
else:
    print(f"Alias computed: mean={adf.df['x_squared'].mean():.3f}")
print()

# =============================================================================
# Example 7: Full stats access
# =============================================================================

print("=" * 60)
print("Example 7: Accessing all stats")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('z')
    print(f"""
Statistics for 'z' (exponential distribution):
  Count:   {stats['n']:,}
  Mean:    {stats['mean']:.4f}
  Std:     {stats['std']:.4f}
  Min:     {stats['min']:.4f}
  Max:     {stats['max']:.4f}
""")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 60)
print("Summary")
print("=" * 60)
print("""
Key points:
1. adf.draw('column') creates a histogram
2. adf.draw('x**2') evaluates expressions on-the-fly
3. Returns (fig, ax, stats) tuple
4. stats dict has: n, mean, std, min, max
5. Customize with bins= and range=
6. For aliases: materialize first, then draw

Next: See scatter_profile.py for 2D plots
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
