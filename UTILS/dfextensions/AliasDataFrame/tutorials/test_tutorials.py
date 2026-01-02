#!/usr/bin/env python3
"""
Tutorial Test Script - Python Version

Tests all tutorials in sequence to verify they work correctly.

Usage:
    cd /path/to/AliasDataFrame
    python tutorials/test_tutorials.py

    # Or with verbose output:
    python tutorials/test_tutorials.py -v
"""

import os
import sys
import subprocess
import argparse

# =============================================================================
# Configuration
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ADF_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'synthetic_data.root')

# Add ADF to path
sys.path.insert(0, ADF_DIR)

# =============================================================================
# Test Functions
# =============================================================================

def print_header(step, title):
    print("=" * 60)
    print(f"Step {step}: {title}")
    print("=" * 60)

def check_prerequisites():
    """Step 0: Check all prerequisites."""
    print_header(0, "Checking prerequisites")
    
    errors = []
    
    # Check AliasDataFrame
    try:
        from AliasDataFrame import AliasDataFrame
        print("✓ AliasDataFrame available")
    except ImportError as e:
        errors.append(f"AliasDataFrame not found: {e}")
        print(f"✗ AliasDataFrame not found")
    
    # Check uproot
    try:
        import uproot
        print("✓ uproot available")
    except ImportError:
        errors.append("uproot not found. Install with: pip install uproot")
        print("✗ uproot not found")
    
    # Check numpy/pandas
    try:
        import numpy
        print("✓ numpy available")
    except ImportError:
        errors.append("numpy not found")
    
    try:
        import pandas
        print("✓ pandas available")
    except ImportError:
        errors.append("pandas not found")
    
    # Check dfdraw (optional)
    try:
        from dfextensions.dfdraw import DFDraw
        print("✓ dfdraw available")
    except ImportError:
        print("⚠ dfdraw not found (tutorials will run without plots)")
    
    print()
    
    if errors:
        print("Errors found:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    return True

def generate_data(force=False):
    """Step 1: Generate synthetic data."""
    print_header(1, "Generating synthetic data")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if os.path.exists(DATA_FILE) and not force:
        print(f"Data file exists: {DATA_FILE}")
        print("Skipping generation (use --force to regenerate)")
        print()
        return True
    
    print(f"Generating: {DATA_FILE}")
    
    generator = os.path.join(ADF_DIR, 'benchmarks', 'generate_synthetic_data.py')
    
    result = subprocess.run([
        sys.executable, generator,
        '-o', DATA_FILE,
        '--rows', '10000',
        '--tracks', '1000',
        '--verify'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✓ Data generation successful")
        print()
        return True
    else:
        print("✗ Data generation failed")
        print(result.stderr)
        return False

def run_tutorial(name, script_path):
    """Run a single tutorial script."""
    print_header(name, f"Testing {os.path.basename(script_path)}")
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(script_path)
    )
    
    # Print output
    if result.stdout:
        print(result.stdout)
    
    if result.returncode == 0:
        print(f"✓ {os.path.basename(script_path)} passed")
        print()
        return True
    else:
        print(f"✗ {os.path.basename(script_path)} failed")
        if result.stderr:
            print("Error output:")
            print(result.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description="Test AliasDataFrame tutorials")
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--force', action='store_true', help='Force data regeneration')
    parser.add_argument('--skip-data', action='store_true', help='Skip data generation')
    args = parser.parse_args()
    
    print("=" * 60)
    print("AliasDataFrame Tutorial Test Suite")
    print("=" * 60)
    print()
    print(f"Tutorial directory: {SCRIPT_DIR}")
    print(f"AliasDataFrame directory: {ADF_DIR}")
    print()
    
    results = {}
    
    # Step 0: Prerequisites
    results['prerequisites'] = check_prerequisites()
    if not results['prerequisites']:
        print("Prerequisites check failed. Aborting.")
        return 1
    
    # Step 1: Generate data
    if not args.skip_data:
        results['data_generation'] = generate_data(force=args.force)
        if not results['data_generation']:
            print("Data generation failed. Aborting.")
            return 1
    else:
        print_header(1, "Skipping data generation (--skip-data)")
        results['data_generation'] = True
        print()
    
    # Step 2-5: Run tutorials
    tutorials = [
        ('2', os.path.join(SCRIPT_DIR, 'drawing', 'histograms.py')),
        ('3', os.path.join(SCRIPT_DIR, 'drawing', 'scatter_profile.py')),
        ('4', os.path.join(SCRIPT_DIR, 'drawing', 'selections_groupby.py')),
        ('5', os.path.join(SCRIPT_DIR, 'drawing', 'with_subframes.py')),
    ]
    
    for step, script in tutorials:
        script_name = os.path.basename(script)
        results[script_name] = run_tutorial(step, script)
        if not results[script_name]:
            print(f"Tutorial {script_name} failed. Stopping.")
            break
    
    # Summary
    print("=" * 60)
    print("TUTORIAL TEST SUMMARY")
    print("=" * 60)
    print()
    
    all_passed = True
    for name, passed in results.items():
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    print()
    
    if all_passed:
        print("=" * 60)
        print("ALL TUTORIALS PASSED")
        print("=" * 60)
        return 0
    else:
        print("=" * 60)
        print("SOME TUTORIALS FAILED")
        print("=" * 60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
