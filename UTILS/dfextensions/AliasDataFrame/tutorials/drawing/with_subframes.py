#!/usr/bin/env python3
"""
AliasDataFrame Tutorial: Drawing with Subframes

Topics: Subframe registration, calibration joins, drawing corrected data
Data: ROOT files from generate_synthetic_data.py
Phase: 6.8 (dfdraw integration with subframes)

Prerequisites:
    1. Generate synthetic data:
       python benchmarks/generate_synthetic_data.py -o tutorials/data/synthetic_data.root

Run:
    python with_subframes.py

Expected output:
    Demonstrates calibration workflow: raw → corrected via subframe join
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from AliasDataFrame import AliasDataFrame

# Check dependencies
try:
    from dfdraw import DFDraw
    HAS_DFDRAW = True
except ImportError:
    HAS_DFDRAW = False
    print("Note: dfdraw not installed. Stats will be shown, plots skipped.")

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
    print("ERROR: uproot required. Install with: pip install uproot")
    sys.exit(1)

# =============================================================================
# Setup: Paths
# =============================================================================

# Data file location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
DATA_FILE = os.path.join(DATA_DIR, 'synthetic_data.root')

# Check if data exists
if not os.path.exists(DATA_FILE):
    print("=" * 60)
    print("DATA FILE NOT FOUND")
    print("=" * 60)
    print(f"Expected: {DATA_FILE}")
    print()
    print("Generate it with:")
    print(f"  python benchmarks/generate_synthetic_data.py -o {DATA_FILE}")
    print()
    print("Or using fallback inline data for demo...")
    print("=" * 60)
    USE_FALLBACK = True
else:
    USE_FALLBACK = False

# =============================================================================
# Load Data
# =============================================================================

print("=" * 60)
print("Loading data...")
print("=" * 60)

if USE_FALLBACK:
    # Create fallback data inline
    np.random.seed(42)
    n_rows = 5000
    n_sectors = 36
    
    # Main data
    df_main = pd.DataFrame({
        'signal': np.abs(np.random.randn(n_rows) * 50 + 100),
        'sec': np.random.randint(0, n_sectors, n_rows),
        'x': np.random.randn(n_rows) * 100 + 200,
    })
    
    # Calibration subframe: gain varies by sector
    df_calib = pd.DataFrame({
        'sec': np.arange(n_sectors),
        'gain': 1.0 + 0.01 * np.arange(n_sectors),  # Known relationship
        'offset': np.random.randn(n_sectors) * 0.1,
    })
    
    adf = AliasDataFrame(df_main)
    calib_adf = AliasDataFrame(df_calib)
    
    print(f"Using fallback inline data")
    print(f"Main data: {len(df_main):,} rows")
    print(f"Calibration: {len(df_calib)} sectors")
    
else:
    # Load from ROOT file (lazy mode)
    adf = AliasDataFrame.read_tree_lazy(DATA_FILE, 'tree')
    print(f"Loaded main tree from: {DATA_FILE}")
    print(f"Available branches: {adf.available_branches[:5]}...")
    
    # Load calibration subframe
    with uproot.open(DATA_FILE) as f:
        calib_data = f['SectorCalib'].arrays(library='pd')
    calib_adf = AliasDataFrame(calib_data)
    print(f"Loaded calibration: {len(calib_data)} sectors")

print()

# =============================================================================
# Example 1: Register subframe
# =============================================================================

print("=" * 60)
print("Example 1: Register calibration subframe")
print("=" * 60)

# Register calibration table with sector as the join key
adf.register_subframe('SectorCalib', calib_adf, index_columns='sec')

print(f"adf.register_subframe('SectorCalib', calib_adf, index_columns='sec')")
print(f"Subframe registered with join key: 'sec'")
print(f"Calibration columns available: {list(calib_adf.df.columns)}")
print()

# =============================================================================
# Example 2: Define alias using subframe
# =============================================================================

print("=" * 60)
print("Example 2: Define calibration alias")
print("=" * 60)

# Apply gain correction using subframe
adf.add_alias('corrected', 'signal * SectorCalib.gain')

print(f"adf.add_alias('corrected', 'signal * SectorCalib.gain')")
print(f"Alias defined. Will join on 'sec' when evaluated.")
print()

# =============================================================================
# Example 3: Draw raw signal
# =============================================================================

print("=" * 60)
print("Example 3: Draw raw signal")
print("=" * 60)

if not USE_FALLBACK:
    adf.ensure_branches(['signal', 'sec'])

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('signal', bins=50)
    print(f"Raw signal stats:")
    print(f"  n={stats['n']:,}, mean={stats['mean']:.2f}, std={stats['std']:.2f}")
else:
    print(f"Raw signal: mean={adf.df['signal'].mean():.2f}")
print()

# =============================================================================
# Example 4: Draw corrected signal
# =============================================================================

print("=" * 60)
print("Example 4: Draw corrected signal")
print("=" * 60)

if HAS_DFDRAW:
    # draw() automatically materializes the alias and joins subframe
    fig, ax, stats = adf.draw('corrected', bins=50)
    print(f"Corrected signal stats:")
    print(f"  n={stats['n']:,}, mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    print(f"  (Mean should be ~3-5% higher due to gain > 1)")
else:
    adf.materialize_alias('corrected')
    print(f"Corrected signal: mean={adf.df['corrected'].mean():.2f}")
print()

# =============================================================================
# Example 5: Profile by sector (QA plot)
# =============================================================================

print("=" * 60)
print("Example 5: Profile plot by sector (QA)")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('corrected:sec', type='profile', bins=36)
    print(f"adf.draw('corrected:sec', type='profile', bins=36)")
    print(f"Shows mean corrected signal per sector")
    print(f"Useful for calibration QA")
print()

# =============================================================================
# Example 6: Compare raw vs corrected
# =============================================================================

print("=" * 60)
print("Example 6: Compare raw vs corrected")
print("=" * 60)

# Define ratio alias
adf.add_alias('correction_factor', 'corrected / signal')

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('correction_factor:sec', type='profile', bins=36)
    print(f"Correction factor vs sector")
    print(f"Should show: gain = 1.0 + 0.01 * sector")
    adf.materialize_alias('correction_factor')
    print(f"Mean correction: {adf.df['correction_factor'].mean():.4f}")
else:
    adf.materialize_alias('correction_factor')
    print(f"Mean correction factor: {adf.df['correction_factor'].mean():.4f}")
print()

# =============================================================================
# Example 7: Selection with subframe alias
# =============================================================================

print("=" * 60)
print("Example 7: Selection with corrected values")
print("=" * 60)

if HAS_DFDRAW:
    fig, ax, stats = adf.draw('corrected', 
                              selection='sec < 18',  # First half of sectors
                              bins=50)
    print(f"selection='sec < 18' (first 18 sectors)")
    print(f"Entries: {stats['n']:,}")
print()

# =============================================================================
# Example 8: Group by with subframe
# =============================================================================

print("=" * 60)
print("Example 8: Group by sector ranges")
print("=" * 60)

# Add sector group alias
adf.add_alias('sector_group', 'sec // 12')  # Groups: 0, 1, 2

if HAS_DFDRAW:
    adf.materialize_alias('sector_group')  # Need to materialize for group_by
    fig, ax, stats = adf.draw('corrected', 
                              group_by='sector_group',
                              bins=50)
    print(f"group_by='sector_group' (sec // 12)")
    print(f"Overlaid histograms for sector groups 0, 1, 2")
print()

# =============================================================================
# Example 9: Verify calibration invariant
# =============================================================================

print("=" * 60)
print("Example 9: Verify calibration invariant")
print("=" * 60)

# The synthetic data has: gain[sec] = 1.0 + 0.01 * sec
# So: corrected = signal * (1.0 + 0.01 * sec)

if 'signal' in adf.df.columns and 'sec' in adf.df.columns:
    expected_gain = 1.0 + 0.01 * adf.df['sec'].values
    expected_corrected = adf.df['signal'].values * expected_gain
    
    if 'corrected' in adf.df.columns:
        actual_corrected = adf.df['corrected'].values
        max_diff = np.abs(actual_corrected - expected_corrected).max()
        print(f"Max difference from expected: {max_diff:.6f}")
        if max_diff < 1e-5:
            print("✓ Calibration invariant verified!")
        else:
            print("⚠ Calibration mismatch detected")
print()

# =============================================================================
# Summary
# =============================================================================

print("=" * 60)
print("Summary")
print("=" * 60)
print("""
Subframe workflow:
1. Register: adf.register_subframe('Name', calib_adf, index_columns='key')
2. Define:   adf.add_alias('corrected', 'signal * Name.gain')
3. Draw:     adf.draw('corrected')  # Joins automatically!

Key points:
- Subframe columns accessed as: SubframeName.column
- Join happens automatically on draw() or materialize_alias()
- Missing keys → NaN (by design, not error)
- Profile plots great for calibration QA

Lazy subframes (alternative):
  adf.register_subframe_lazy('Name', 'file.root', 'tree', index_columns='key')
  # Loads only when needed

Next: See workflows/ for complete calibration pipeline examples
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
