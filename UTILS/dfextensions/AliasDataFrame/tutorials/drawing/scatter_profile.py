#!/usr/bin/env python3
"""
AliasDataFrame Tutorial: Scatter and Profile Plots

Topics: 2D scatter, profile plots, hist2d, hexbin, color mapping
Data: Inline NumPy (no file dependencies)
Phase: 6.8 (dfdraw integration)

Run:
    python scatter_profile.py

Expected output:
    Statistics for each 2D plot variant
    Demonstrates y:x syntax for 2D plots
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory for imports
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
# Setup: Create sample data with correlations
# =============================================================================

print("=" * 60)
print("Setting up sample data with correlations...")
print("=" * 60)

np.random.seed(42)
n_rows = 5000

# Create correlated data
x = np.random.randn(n_rows)
y = 2 * x + np.random.randn(n_rows) * 0.5  # Linear relationship with noise
z = x**2 + np.random.randn(n_rows) * 0.2   # Quadratic relationship
sector = np.random.randint(0, 18, n_rows)  # Discrete sectors

df = pd.DataFrame({
    'x': x,
    'y': y,
    'z': z,
    'sector': sector,
    'weight': np.abs(np.random.randn(n_rows))  # For color mapping
})

adf = AliasDataFrame(df)
print(f"Created AliasDataFrame with {len(adf.df):,} rows")
print(f"Correlations: y ≈ 2*x, z ≈ x²")
print()

# =============================================================================
# Example 1: Basic 2D scatter
# =============================================================================

print("=" * 60)
print("Example 1: Basic 2D scatter")
print("=" * 60)

if HAS_DFDRAW:
    # Syntax: 'y:x' means plot y vs x (y on vertical axis)
    fig, ax, stats = adf.draw('y:x')
    print(f"Syntax: adf.draw('y:x')  # y vs x")
    print(f"Entries: {stats['n']:,}")
else:
    print("Syntax: adf.draw('y:x')  # y vs x")
    print(f"Would plot y vs x scatter")
print()

# =============================================================================
# Example 2: Scatter with explicit method
# =============================================================================

print("=" * 60)
print("Example 2: Using .scatter() method")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.scatter('y:x')
    print(f"adf.scatter('y:x') - same result as draw('y:x')")
    print(f"Stats: n={stats['n']:,}")
print()

# =============================================================================
# Example 3: Profile plot (mean y vs x)
# =============================================================================

print("=" * 60)
print("Example 3: Profile plot (mean y vs binned x)")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('y:x', type='profile', bins=50)
    print(f"adf.draw('y:x', type='profile', bins=50)")
    print(f"Shows mean y in each x bin")
    print(f"Stats keys: {list(stats.keys())}")
    print(f"Entries: n={stats.get('n', 'N/A')}")
    print(f"Expected slope ≈ 2.0 (since y = 2*x + noise)")
print()

# =============================================================================
# Example 4: Profile with explicit method
# =============================================================================

print("=" * 60)
print("Example 4: Using .profile() method")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.profile('z:x', bins=30)
    print(f"adf.profile('z:x', bins=30)")
    print(f"Shows mean z vs x (should show parabola since z ≈ x²)")
print()

# =============================================================================
# Example 5: 2D histogram
# =============================================================================

print("=" * 60)
print("Example 5: 2D histogram (heatmap)")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('y:x', type='hist2d', bins=50)
    print(f"adf.draw('y:x', type='hist2d', bins=50)")
    print(f"Color shows density of points")
    print(f"Stats: n={stats['n']:,}")
print()

# =============================================================================
# Example 6: hist2d with different bin counts per axis
# =============================================================================

print("=" * 60)
print("Example 6: hist2d with asymmetric bins")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.hist2d('y:x', bins=[100, 50])
    print(f"adf.hist2d('y:x', bins=[100, 50])")
    print(f"100 bins in x, 50 bins in y")
print()

# =============================================================================
# Example 7: Hexbin plot
# =============================================================================

print("=" * 60)
print("Example 7: Hexbin plot")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.hexbin('y:x', gridsize=30)
    print(f"adf.hexbin('y:x', gridsize=30)")
    print(f"Hexagonal binning - good for large datasets")
print()

# =============================================================================
# Example 8: Color mapping (3rd variable)
# =============================================================================

print("=" * 60)
print("Example 8: Color mapping by third variable")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('y:x', color='weight')
    print(f"adf.draw('y:x', color='weight')")
    print(f"Points colored by 'weight' column")
print()

# =============================================================================
# Example 9: Profile by sector (discrete x)
# =============================================================================

print("=" * 60)
print("Example 9: Profile with discrete x (sectors)")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('y:sector', type='profile', bins=18)
    print(f"adf.draw('y:sector', type='profile', bins=18)")
    print(f"Mean y per sector (useful for calibration QA)")
print()

# =============================================================================
# Example 10: Using expressions in 2D plots
# =============================================================================

print("=" * 60)
print("Example 10: 2D plot with expression")
print("=" * 60)

# Compute residual as a new column (simpler than alias for dfdraw)
adf.df['residual'] = adf.df['y'] - 2 * adf.df['x']

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('residual:x', type='profile', bins=50)
    print(f"Residual = y - 2*x (should be ~0)")
    print(f"Mean residual: {adf.df['residual'].mean():.4f}")
else:
    print(f"Mean residual: {adf.df['residual'].mean():.4f}")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 60)
print("Summary")
print("=" * 60)
print("""
Key points:
1. 2D syntax: 'y:x' (y vs x, y on vertical axis)
2. Scatter: adf.draw('y:x') or adf.scatter('y:x')
3. Profile: type='profile' shows mean y per x bin
4. hist2d: type='hist2d' for density heatmap
5. Hexbin: adf.hexbin() for large datasets
6. Color: color='z' colors points by third variable

Plot type summary:
  - scatter:  Individual points
  - profile:  Mean ± error bars
  - hist2d:   Density heatmap
  - hexbin:   Hexagonal density

Next: See selections_groupby.py for filtering and overlays
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
