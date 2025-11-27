#!/usr/bin/env python3
"""
generate_synthetic_data.py - Generate synthetic ROOT file for benchmarks

Creates a small (~5MB) ROOT file with realistic structure for testing
AliasDataFrame functionality without requiring real data.

Usage:
    python generate_synthetic_data.py                    # Default output
    python generate_synthetic_data.py --output data.root # Custom path
    python generate_synthetic_data.py --rows 100000      # Custom size

Output:
    - Main tree with typical TPC-like columns
    - Subframe tree 'T' with track-level data
    - ~5MB file size (100k rows default)
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


def verify_file(filepath):
    """Verify the generated ROOT file can be read."""
    try:
        import uproot
        
        print(f"\nVerifying file...")
        
        with uproot.open(filepath) as f:
            # Check main tree
            tree = f['tree']
            print(f"  ✓ Main tree: {tree.num_entries:,} entries")
            print(f"    Branches: {list(tree.keys())}")
            
            # Check subframe
            T = f['T']
            print(f"  ✓ Subframe T: {T.num_entries:,} entries")
            print(f"    Branches: {list(T.keys())}")
        
        # Test with AliasDataFrame
        try:
            from AliasDataFrame import AliasDataFrame
            
            adf = AliasDataFrame.read_tree(filepath, 'tree', load_subframes=True)
            print(f"  ✓ AliasDataFrame loaded: {len(adf.df):,} rows")
            
            subframes = list(adf._subframes.subframes.keys())
            print(f"    Subframes detected: {subframes}")
            
            return True
            
        except ImportError:
            print("  (Skipped AliasDataFrame verification - not in path)")
            return True
            
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
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
        """
    )
    parser.add_argument('--output', '-o', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'synthetic_data.root'),
                        help='Output file path (default: benchmarks/synthetic_data.root)')
    parser.add_argument('--rows', type=int, default=100_000,
                        help='Number of rows in main tree (default: 100000)')
    parser.add_argument('--tracks', type=int, default=10_000,
                        help='Number of tracks in subframe (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify file after generation')
    
    args = parser.parse_args()
    
    stats = generate_synthetic_root(
        output_path=args.output,
        n_rows=args.rows,
        n_tracks=args.tracks,
        seed=args.seed
    )
    
    if stats is None:
        sys.exit(1)
    
    if args.verify:
        if not verify_file(args.output):
            sys.exit(1)
    
    print(f"\nDone. File ready at: {args.output}")


if __name__ == '__main__':
    main()
