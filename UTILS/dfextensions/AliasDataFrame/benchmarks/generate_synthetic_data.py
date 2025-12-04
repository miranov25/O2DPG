#!/usr/bin/env python3
"""
generate_synthetic_data.py - Generate synthetic ROOT file for benchmarks

Creates a small (~5MB) ROOT file with realistic structure for testing
AliasDataFrame functionality without requiring real data.

Usage:
    python generate_synthetic_data.py                    # Default output
    python generate_synthetic_data.py --output data.root # Custom path
    python generate_synthetic_data.py --rows 100000      # Custom size
    python generate_synthetic_data.py --rdf              # RDF mode (4 subframes)
    python generate_synthetic_data.py --rdf --sparse     # RDF with sparse keys

Output:
    - Main tree with typical TPC-like columns
    - Subframe tree 'T' with track-level data
    - ~5MB file size (100k rows default)
    
RDF Mode (--rdf):
    - 4 subframes: T (1-key), R (1-key), DITS0FitSide (2-key), DTrack0 (3-key)
    - Supports sparse key testing with --sparse flag
"""

import argparse
import os
import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_synthetic_root(output_path, n_rows=100_000, n_tracks=10_000, seed=42):
    """
    Generate synthetic ROOT file with main tree and subframe.
    
    Parameters
    ----------
    output_path : str
        Output ROOT file path
    n_rows : int
        Number of rows in main tree
    n_tracks : int
        Number of unique tracks (for subframe)
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict : Statistics about generated file
    """
    try:
        import uproot
    except ImportError:
        print("ERROR: uproot is required. Install with: pip install uproot")
        return None
    
    np.random.seed(seed)
    
    print(f"Generating synthetic ROOT file: {output_path}")
    print(f"  Main tree rows: {n_rows:,}")
    print(f"  Subframe tracks: {n_tracks:,}")
    
    # =========================================================================
    # Generate main tree data (TPC cluster-like)
    # =========================================================================
    
    # Track indices (for joining with subframe)
    track_idx = np.random.randint(0, n_tracks, n_rows, dtype=np.int32)
    
    # Position columns (float32)
    x = np.random.randn(n_rows).astype(np.float32) * 100 + 200
    y = np.random.randn(n_rows).astype(np.float32) * 10
    z = np.random.randn(n_rows).astype(np.float32) * 200
    
    # Delta columns (float16 - for compression testing)
    dy = np.random.randn(n_rows).astype(np.float16)
    dz = np.random.randn(n_rows).astype(np.float16)
    
    # Sector/row columns (uint8)
    sec = np.random.randint(0, 36, n_rows, dtype=np.uint8)
    row = np.random.randint(0, 152, n_rows, dtype=np.uint8)
    
    # Additional physics columns
    mX = np.random.randn(n_rows).astype(np.float32) * 50
    mY = np.random.randn(n_rows).astype(np.float32) * 50
    
    main_data = {
        'track_idx': track_idx,
        'x': x,
        'y': y,
        'z': z,
        'dy': dy.astype(np.float32),  # uproot needs float32
        'dz': dz.astype(np.float32),
        'sec': sec,
        'row': row,
        'mX': mX,
        'mY': mY,
    }
    
    # =========================================================================
    # Generate subframe data (track-level)
    # =========================================================================
    
    # One row per unique track
    track_ids = np.arange(n_tracks, dtype=np.int32)
    
    # Track parameters
    mP3 = np.random.randn(n_tracks).astype(np.float32) * 0.1
    mP4 = np.random.randn(n_tracks).astype(np.float32) * 0.01
    dEdxTPC = np.random.exponential(50, n_tracks).astype(np.float32)
    
    subframe_data = {
        'track_idx': track_ids,
        'mP3': mP3,
        'mP4': mP4,
        'mX': np.random.randn(n_tracks).astype(np.float32) * 10,
        'dEdxTPC': dEdxTPC,
    }
    
    # =========================================================================
    # Write ROOT file
    # =========================================================================
    
    with uproot.recreate(output_path) as f:
        # Main tree
        f['tree'] = main_data
        
        # Subframe tree
        f['T'] = subframe_data
    
    # Get file size
    file_size = os.path.getsize(output_path)
    
    stats = {
        'path': output_path,
        'size_bytes': file_size,
        'size_mb': file_size / (1024 * 1024),
        'main_rows': n_rows,
        'main_columns': len(main_data),
        'subframe_rows': n_tracks,
        'subframe_columns': len(subframe_data),
    }
    
    print(f"\n✓ Generated successfully:")
    print(f"  File size: {stats['size_mb']:.1f} MB")
    print(f"  Main tree: {n_rows:,} rows × {len(main_data)} columns")
    print(f"  Subframe T: {n_tracks:,} rows × {len(subframe_data)} columns")
    
    return stats


def generate_rdf_synthetic_root(output_path, n_rows=1_000_000, seed=42, sparse_keys=False):
    """
    Generate synthetic ROOT file for RDF benchmarks with 4 subframes.
    
    Uses AliasDataFrame.export_tree() for proper alias/schema embedding.
    
    Creates structure matching real calibration data:
    - Main tree with indices and base columns
    - T: 1-key subframe (entry)
    - R: 1-key subframe (row)
    - DITS0FitSide: 2-key subframe (row, drift)
    - DTrack0: 3-key subframe (entry, row, drift)
    
    Parameters
    ----------
    output_path : str
        Output ROOT file path
    n_rows : int
        Number of rows in main tree
    seed : int
        Random seed for reproducibility
    sparse_keys : bool
        If True, use sparse (non-contiguous) key values for testing
        
    Returns
    -------
    dict : Statistics about generated file
    """
    import pandas as pd
    from AliasDataFrame import AliasDataFrame
    
    np.random.seed(seed)
    
    # Configuration
    n_entries = min(1000, n_rows // 100)
    n_row_vals = 152
    n_drift_vals = 10
    
    print(f"Generating RDF benchmark ROOT file: {output_path}")
    print(f"  Main tree rows: {n_rows:,}")
    print(f"  Unique entries: {n_entries}")
    print(f"  Row values: {n_row_vals}")
    print(f"  Drift values: {n_drift_vals}")
    print(f"  Sparse keys: {sparse_keys}")
    
    # =========================================================================
    # Main DataFrame
    # =========================================================================
    
    entry = np.random.randint(0, n_entries, n_rows, dtype=np.int32)
    row = np.random.randint(0, n_row_vals, n_rows, dtype=np.int32)
    drift = np.random.randint(0, n_drift_vals, n_rows, dtype=np.int32)
    
    if sparse_keys:
        orbit_base = np.array([0, 1_000_000, 5_000_000, 10_000_000, 50_000_000], dtype=np.int64)
        firstTFOrbit = orbit_base[entry % len(orbit_base)] + (entry // len(orbit_base))
    else:
        firstTFOrbit = (entry * 1000).astype(np.int64)
    
    main_df = pd.DataFrame({
        'entry': entry,
        'row': row,
        'drift': drift,
        'firstTFOrbit': firstTFOrbit,
        'tfCounter': (entry % 100).astype(np.int32),
        'y': np.random.randn(n_rows).astype(np.float32),
        'z': np.random.randn(n_rows).astype(np.float32),
        'dy': np.random.randn(n_rows).astype(np.float32) * 0.1,
        'dz': np.random.randn(n_rows).astype(np.float32) * 0.1,
        'mX': np.random.randn(n_rows).astype(np.float32) * 50,
        'mY': np.random.randn(n_rows).astype(np.float32) * 50,
    })
    
    # =========================================================================
    # Subframes
    # =========================================================================
    
    T_df = pd.DataFrame({
        'entry': np.arange(n_entries, dtype=np.int32),
        'mP3': np.random.randn(n_entries).astype(np.float32) * 0.1,
        'mP4': np.random.randn(n_entries).astype(np.float32) * 0.01,
    })
    
    R_df = pd.DataFrame({
        'row': np.arange(n_row_vals, dtype=np.int32),
        'refX': (85.0 + np.arange(n_row_vals) * 1.5).astype(np.float32),
    })
    
    n_dits = n_row_vals * n_drift_vals
    DITS_df = pd.DataFrame({
        'row': np.repeat(np.arange(n_row_vals, dtype=np.int32), n_drift_vals),
        'drift': np.tile(np.arange(n_drift_vals, dtype=np.int32), n_row_vals),
        'yDelta': np.random.randn(n_dits).astype(np.float32) * 0.01,
        'zDelta': np.random.randn(n_dits).astype(np.float32) * 0.01,
    })
    
    # DTrack0: sampled 3-key subframe
    n_dt_samples = min(100_000, n_entries * n_row_vals * n_drift_vals // 100)
    dt_entries = np.random.randint(0, n_entries, n_dt_samples, dtype=np.int32)
    dt_rows = np.random.randint(0, n_row_vals, n_dt_samples, dtype=np.int32)
    dt_drifts = np.random.randint(0, n_drift_vals, n_dt_samples, dtype=np.int32)
    combined = np.column_stack([dt_entries, dt_rows, dt_drifts])
    _, unique_idx = np.unique(combined, axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    
    DTrack_df = pd.DataFrame({
        'entry': dt_entries[unique_idx],
        'row': dt_rows[unique_idx],
        'drift': dt_drifts[unique_idx],
        'corrY': np.random.randn(len(unique_idx)).astype(np.float32) * 0.005,
        'corrZ': np.random.randn(len(unique_idx)).astype(np.float32) * 0.005,
    })
    
    # =========================================================================
    # Compute composite key for DTrack0 (3-key subframe)
    # =========================================================================
    
    from AliasDataFrameRDF import compute_composite_key_sparse
    
    main_keys, sub_keys = compute_composite_key_sparse(
        main_df, DTrack_df, ['entry', 'row', 'drift']
    )
    main_df['__adf_key_DTrack0__'] = main_keys
    DTrack_df['__adf_key_DTrack0__'] = sub_keys
    
    print(f"  Composite key for DTrack0: {len(np.unique(sub_keys))} unique values")
    
    # =========================================================================
    # Create AliasDataFrame
    # =========================================================================
    
    adf = AliasDataFrame(main_df)
    
    # Create subframe AliasDataFrames and register them
    adf.register_subframe('T', AliasDataFrame(T_df), index_columns=['entry'])
    adf.register_subframe('R', AliasDataFrame(R_df), index_columns=['row'])
    adf.register_subframe('DITS0FitSide', AliasDataFrame(DITS_df), index_columns=['row', 'drift'])
    adf.register_subframe('DTrack0', AliasDataFrame(DTrack_df), index_columns=['entry', 'row', 'drift'])
    
    # =========================================================================
    # Add aliases - simple
    # =========================================================================
    
    adf.add_alias('ySquared', 'y**2', dtype=np.float32)
    adf.add_alias('zSquared', 'z**2', dtype=np.float32)
    adf.add_alias('radius', 'np.sqrt(y**2 + z**2)', dtype=np.float32)
    
    # Add aliases - with subframe joins
    adf.add_alias('dyC1', 'dy - T.mP3', dtype=np.float32)
    adf.add_alias('dzC1', 'dz - T.mP4', dtype=np.float32)
    adf.add_alias('dyC2', 'dy - T.mP3 - DITS0FitSide.yDelta', dtype=np.float32)
    adf.add_alias('dzC2', 'dz - T.mP4 - DITS0FitSide.zDelta', dtype=np.float32)
    adf.add_alias('xRef', 'R.refX', dtype=np.float32)
    adf.add_alias('correctedY', 'y - dy + T.mP3 + DITS0FitSide.yDelta', dtype=np.float32)
    
    # =========================================================================
    # Add aliases - 10-level chain
    # =========================================================================
    
    adf.add_alias('L1', 'y + z', dtype=np.float32)
    adf.add_alias('L2', 'L1 * T.mP3', dtype=np.float32)
    adf.add_alias('L3', 'L2 - R.refX', dtype=np.float32)
    adf.add_alias('L4', 'L3 + DITS0FitSide.yDelta', dtype=np.float32)
    adf.add_alias('L5', 'np.sqrt(L4**2)', dtype=np.float32)
    adf.add_alias('L6', 'L5 * 2', dtype=np.float32)
    adf.add_alias('L7', 'L6 - L1', dtype=np.float32)
    adf.add_alias('L8', 'np.abs(L7)', dtype=np.float32)
    adf.add_alias('L9', 'L8 + L3', dtype=np.float32)
    adf.add_alias('L10', 'L9 / 2', dtype=np.float32)
    
    # =========================================================================
    # Add aliases - 20-level chain (extends from L10)
    # =========================================================================
    
    adf.add_alias('L11', 'L10 + T.mP4', dtype=np.float32)
    adf.add_alias('L12', 'L11 - L2', dtype=np.float32)
    adf.add_alias('L13', 'L12 * 1.5', dtype=np.float32)
    adf.add_alias('L14', 'L13 + L5', dtype=np.float32)
    adf.add_alias('L15', 'np.abs(L14)', dtype=np.float32)
    adf.add_alias('L16', 'L15 - L8', dtype=np.float32)
    adf.add_alias('L17', 'L16 / 3', dtype=np.float32)
    adf.add_alias('L18', 'L17 + L10', dtype=np.float32)
    adf.add_alias('L19', 'L18 * 2', dtype=np.float32)
    adf.add_alias('L20', 'L19 - L5', dtype=np.float32)
    
    # =========================================================================
    # Materialize key aliases and store as ground truth (_mat suffix)
    # =========================================================================
    
    print("\n  Materializing ground truth...")
    
    # Materialize key aliases
    adf.materialize_alias('dyC2')
    adf.materialize_alias('dzC2')
    adf.materialize_alias('L10')
    adf.materialize_alias('L20')
    
    # Copy to _mat columns (ground truth)
    adf.df['dyC2_mat'] = adf.df['dyC2'].copy()
    adf.df['dzC2_mat'] = adf.df['dzC2'].copy()
    adf.df['L10_mat'] = adf.df['L10'].copy()
    adf.df['L20_mat'] = adf.df['L20'].copy()
    
    print(f"    dyC2_mat: mean={adf.df['dyC2_mat'].mean():.6f}")
    print(f"    L10_mat:  mean={adf.df['L10_mat'].mean():.6f}")
    print(f"    L20_mat:  mean={adf.df['L20_mat'].mean():.6f}")
    
    # =========================================================================
    # Export using AliasDataFrame (embeds aliases, schema, subframes properly)
    # =========================================================================
    
    adf.export_tree(output_path, 'tree')
    
    # Get file size
    file_size = os.path.getsize(output_path)
    
    stats = {
        'path': output_path,
        'size_bytes': file_size,
        'size_mb': file_size / (1024 * 1024),
        'main_rows': n_rows,
        'main_columns': len(main_df.columns),
        'subframes': {
            'T': {'rows': len(T_df), 'keys': ['entry']},
            'R': {'rows': len(R_df), 'keys': ['row']},
            'DITS0FitSide': {'rows': len(DITS_df), 'keys': ['row', 'drift']},
            'DTrack0': {'rows': len(DTrack_df), 'keys': ['entry', 'row', 'drift']},
        },
        'aliases': list(adf.aliases.keys()),
        'ground_truth': ['dyC2_mat', 'dzC2_mat', 'L10_mat', 'L20_mat'],
        'sparse_keys': sparse_keys,
    }
    
    print(f"\n✓ Generated successfully:")
    print(f"  File size: {stats['size_mb']:.1f} MB")
    print(f"  Main tree: {n_rows:,} rows × {len(main_df.columns)} columns")
    print(f"  Subframe T: {len(T_df):,} rows (1-key: entry)")
    print(f"  Subframe R: {len(R_df):,} rows (1-key: row)")
    print(f"  Subframe DITS0FitSide: {len(DITS_df):,} rows (2-key: row, drift)")
    print(f"  Subframe DTrack0: {len(DTrack_df):,} rows (3-key: entry, row, drift)")
    print(f"\n  Aliases ({len(stats['aliases'])}): L1-L20, dyC1/2, dzC1/2, etc.")
    print(f"  Ground truth: {stats['ground_truth']}")
    
    return stats


def verify_file(filepath, rdf_mode=False):
    """Verify the generated ROOT file can be read."""
    try:
        print(f"\nVerifying file...")
        
        # Test with AliasDataFrame
        from AliasDataFrame import AliasDataFrame
        
        adf = AliasDataFrame.read_tree(filepath, 'tree', load_subframes=True)
        print(f"  ✓ AliasDataFrame loaded: {len(adf.df):,} rows")
        print(f"    Columns: {list(adf.df.columns)}")
        
        subframes = list(adf._subframes.subframes.keys()) if hasattr(adf, '_subframes') else []
        print(f"    Subframes: {subframes}")
        
        aliases = list(adf.aliases.keys()) if hasattr(adf, 'aliases') else []
        if aliases:
            print(f"    Aliases ({len(aliases)}): {aliases}")
        elif rdf_mode:
            print(f"  ⚠ Warning: No aliases found in RDF mode file")
        
        return True
            
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ROOT file for benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python generate_synthetic_data.py
    python generate_synthetic_data.py --output test_data.root
    python generate_synthetic_data.py --rows 500000 --tracks 50000
    
RDF Benchmark Mode:
    python generate_synthetic_data.py --rdf
    python generate_synthetic_data.py --rdf --rows 1000000
    python generate_synthetic_data.py --rdf --sparse  # Test sparse key algorithm
        """
    )
    parser.add_argument('--output', '-o', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'synthetic_data.root'),
                        help='Output file path (default: benchmarks/synthetic_data.root)')
    parser.add_argument('--rows', type=int, default=100_000,
                        help='Number of rows in main tree (default: 100000)')
    parser.add_argument('--tracks', type=int, default=10_000,
                        help='Number of tracks in subframe (default: 10000, ignored in --rdf mode)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify file after generation')
    parser.add_argument('--rdf', action='store_true',
                        help='Generate RDF benchmark data (4 subframes, multi-key indices)')
    parser.add_argument('--sparse', action='store_true',
                        help='Use sparse (non-contiguous) key values (only with --rdf)')
    
    args = parser.parse_args()
    
    if args.rdf:
        # RDF mode: 4 subframes with multi-key indices
        stats = generate_rdf_synthetic_root(
            output_path=args.output,
            n_rows=args.rows,
            seed=args.seed,
            sparse_keys=args.sparse
        )
    else:
        # Standard mode: main tree + T subframe
        stats = generate_synthetic_root(
            output_path=args.output,
            n_rows=args.rows,
            n_tracks=args.tracks,
            seed=args.seed
        )
    
    if stats is None:
        sys.exit(1)
    
    if args.verify:
        if not verify_file(args.output, rdf_mode=args.rdf):
            sys.exit(1)
    
    print(f"\nDone. File ready at: {args.output}")


if __name__ == '__main__':
    main()
