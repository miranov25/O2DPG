#!/usr/bin/env python3
"""
generate_synthetic_data.py - Generate synthetic ROOT files for benchmarks and testing

Creates synthetic ROOT files with realistic structure for testing AliasDataFrame
functionality without requiring real data.

Usage:
    python generate_synthetic_data.py                    # Default: single file
    python generate_synthetic_data.py --output data.root # Custom path
    python generate_synthetic_data.py --rows 100000      # Custom size
    python generate_synthetic_data.py --rdf              # RDF mode (4 subframes)
    python generate_synthetic_data.py --rdf --sparse     # RDF with sparse keys

Chain Mode (Phase 6.8):
    python generate_synthetic_data.py --chain 5 -o chain_data/
        → Creates data_run001.root ... data_run005.root
    
    python generate_synthetic_data.py --chain 5 --subframe-chain 3 -o chain_data/
        → Creates data_run001-005.root + calib_001-003.root

The generated data has KNOWN RELATIONSHIPS for invariance testing:
    - y_derived = 2 * x (exact linear relationship)
    - corrected = signal * gain[sector] (calibration join)
    - gain varies by sector (deterministic per seed)

Output:
    - Main tree with typical TPC-like columns
    - Subframe tree 'T' with track-level calibration data
    - ~5MB file size per 100k rows
    
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

# Default configuration
DEFAULT_ROWS = 100_000
DEFAULT_TRACKS = 10_000
RNG_SEED = 42

# Number of sectors (for calibration joins)
N_SECTORS = 36


def generate_synthetic_root(output_path, n_rows=100_000, n_tracks=10_000, seed=42,
                           file_index=0, include_derived=True):
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
        Base random seed for reproducibility
    file_index : int
        File index for chain generation (affects seed)
    include_derived : bool
        If True, include y_derived = 2*x for invariance testing
        
    Returns
    -------
    dict : Statistics about generated file
    """
    try:
        import uproot
    except ImportError:
        print("ERROR: uproot is required. Install with: pip install uproot")
        return None
    
    # Deterministic seed based on file index
    actual_seed = seed + file_index * 1000
    np.random.seed(actual_seed)
    
    print(f"Generating synthetic ROOT file: {output_path}")
    print(f"  Seed: {actual_seed} (base={seed}, file_index={file_index})")
    print(f"  Main tree rows: {n_rows:,}")
    print(f"  Subframe tracks: {n_tracks:,}")
    
    # =========================================================================
    # Generate main tree data (TPC cluster-like)
    # =========================================================================
    
    # Track indices (for joining with subframe)
    track_idx = np.random.randint(0, n_tracks, n_rows, dtype=np.int32)
    
    # Base position column - x is the source for y_derived
    x = np.random.randn(n_rows).astype(np.float32) * 100 + 200
    
    # y_derived = 2 * x (EXACT relationship for invariance testing)
    # NO noise - this allows np.array_equal in tests
    y_derived = (2.0 * x).astype(np.float32)
    
    # Other position columns
    y = np.random.randn(n_rows).astype(np.float32) * 10
    z = np.random.randn(n_rows).astype(np.float32) * 200
    
    # Signal column (for calibration testing)
    signal = np.abs(np.random.randn(n_rows).astype(np.float32) * 50 + 100)
    
    # Sector/row columns
    sec = np.random.randint(0, N_SECTORS, n_rows, dtype=np.int32)
    row = np.random.randint(0, 152, n_rows, dtype=np.int32)
    
    # File index column (for chain testing - identifies which file the data came from)
    file_idx_col = np.full(n_rows, file_index, dtype=np.int32)
    
    # Run number (varies by file for chain calibration testing)
    run_number = np.full(n_rows, 1000 + file_index, dtype=np.int32)
    
    main_data = {
        'track_idx': track_idx,
        'x': x,
        'y': y,
        'z': z,
        'signal': signal,
        'sec': sec,
        'row': row,
        'file_idx': file_idx_col,
        'run_number': run_number,
    }
    
    # Include derived column for invariance testing
    if include_derived:
        main_data['y_derived'] = y_derived
    
    # =========================================================================
    # Generate subframe data (track-level calibration)
    # =========================================================================
    
    # One row per unique track
    track_ids = np.arange(n_tracks, dtype=np.int32)
    
    # Track parameters for calibration
    mP3 = np.random.randn(n_tracks).astype(np.float32) * 0.1
    mP4 = np.random.randn(n_tracks).astype(np.float32) * 0.01
    dEdxTPC = np.random.exponential(50, n_tracks).astype(np.float32)
    
    subframe_data = {
        'track_idx': track_ids,
        'mP3': mP3,
        'mP4': mP4,
        'dEdxTPC': dEdxTPC,
    }
    
    # =========================================================================
    # Generate sector calibration subframe (for sector-based joins)
    # Gain is deterministic per sector for invariance testing
    # =========================================================================
    
    sector_ids = np.arange(N_SECTORS, dtype=np.int32)
    # Deterministic gain: gain[sector] = 1.0 + 0.01 * sector
    # This allows exact verification: corrected = signal * (1.0 + 0.01 * sec)
    gain = (1.0 + 0.01 * sector_ids).astype(np.float32)
    offset = (np.random.randn(N_SECTORS) * 0.1).astype(np.float32)
    
    sector_calib_data = {
        'sec': sector_ids,
        'gain': gain,
        'offset': offset,
    }
    
    # =========================================================================
    # Write ROOT file
    # =========================================================================
    
    with uproot.recreate(output_path) as f:
        # Main tree
        f['tree'] = main_data
        
        # Track subframe
        f['T'] = subframe_data
        
        # Sector calibration subframe
        f['SectorCalib'] = sector_calib_data
    
    # Get file size
    file_size = os.path.getsize(output_path)
    
    stats = {
        'path': output_path,
        'size_bytes': file_size,
        'size_mb': file_size / (1024 * 1024),
        'main_rows': n_rows,
        'main_columns': len(main_data),
        'subframe_T_rows': n_tracks,
        'subframe_T_columns': len(subframe_data),
        'subframe_SectorCalib_rows': N_SECTORS,
        'file_index': file_index,
        'seed': actual_seed,
    }
    
    print(f"  ✓ Generated successfully: {stats['size_mb']:.1f} MB")
    
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


def generate_calibration_file(output_path, seed=42, file_index=0, n_runs=10):
    """
    Generate a calibration file for subframe chain testing.
    
    The calibration contains run-dependent gain values for sector calibration.
    
    Parameters
    ----------
    output_path : str
        Output ROOT file path
    seed : int
        Base random seed
    file_index : int
        File index for chain (affects seed and run range)
    n_runs : int
        Number of runs covered by this calibration file
        
    Returns
    -------
    dict : Statistics about generated file
    """
    try:
        import uproot
    except ImportError:
        print("ERROR: uproot is required. Install with: pip install uproot")
        return None
    
    # Deterministic seed based on file index
    actual_seed = seed + file_index * 500 + 10000
    np.random.seed(actual_seed)
    
    print(f"Generating calibration file: {output_path}")
    print(f"  Seed: {actual_seed}")
    
    # Run numbers covered by this calibration file
    # Each file covers a range of runs
    run_start = 1000 + file_index * n_runs
    run_end = run_start + n_runs
    
    # Create calibration entries for each (run, sector) combination
    n_entries = n_runs * N_SECTORS
    
    runs = np.repeat(np.arange(run_start, run_end, dtype=np.int32), N_SECTORS)
    sectors = np.tile(np.arange(N_SECTORS, dtype=np.int32), n_runs)
    
    # Gain varies by run and sector (deterministic)
    # gain[run, sector] = 1.0 + 0.01 * sector + 0.001 * (run - 1000)
    gain = (1.0 + 0.01 * sectors + 0.001 * (runs - 1000)).astype(np.float32)
    offset = (np.random.randn(n_entries) * 0.05).astype(np.float32)
    
    calib_data = {
        'run_number': runs,
        'sec': sectors,
        'gain': gain,
        'offset': offset,
    }
    
    with uproot.recreate(output_path) as f:
        f['tree'] = calib_data
    
    file_size = os.path.getsize(output_path)
    
    stats = {
        'path': output_path,
        'size_bytes': file_size,
        'size_mb': file_size / (1024 * 1024),
        'rows': n_entries,
        'run_range': (run_start, run_end - 1),
        'file_index': file_index,
        'seed': actual_seed,
    }
    
    print(f"  ✓ Generated: {n_entries} entries, runs {run_start}-{run_end-1}")
    
    return stats


def generate_chain_data(output_dir, n_chain_files=5, n_subframe_chain_files=0,
                       rows_per_file=100_000, tracks_per_file=10_000, seed=42):
    """
    Generate multiple files for chain testing.
    
    Parameters
    ----------
    output_dir : str
        Output directory for all files
    n_chain_files : int
        Number of main data files to generate
    n_subframe_chain_files : int
        Number of calibration subframe files to generate
    rows_per_file : int
        Rows per main data file
    tracks_per_file : int
        Tracks per main data file
    seed : int
        Base random seed
        
    Returns
    -------
    dict : Statistics about all generated files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Generating chain data in: {output_dir}")
    print(f"  Main data files: {n_chain_files}")
    print(f"  Subframe chain files: {n_subframe_chain_files}")
    print(f"  Rows per file: {rows_per_file:,}")
    print(f"  Base seed: {seed}")
    print(f"{'='*60}\n")
    
    stats = {
        'output_dir': output_dir,
        'main_files': [],
        'subframe_files': [],
        'total_rows': 0,
        'total_size_mb': 0,
    }
    
    # Generate main data files
    for i in range(n_chain_files):
        filename = f"data_run{i+1:03d}.root"
        filepath = os.path.join(output_dir, filename)
        
        file_stats = generate_synthetic_root(
            filepath,
            n_rows=rows_per_file,
            n_tracks=tracks_per_file,
            seed=seed,
            file_index=i,
            include_derived=True
        )
        
        if file_stats:
            stats['main_files'].append(file_stats)
            stats['total_rows'] += file_stats['main_rows']
            stats['total_size_mb'] += file_stats['size_mb']
    
    # Generate calibration subframe chain files
    for i in range(n_subframe_chain_files):
        filename = f"calib_{i+1:03d}.root"
        filepath = os.path.join(output_dir, filename)
        
        # Each calibration file covers runs for n_chain_files // n_subframe_chain_files files
        # This creates overlapping coverage for testing
        runs_per_calib = max(1, (n_chain_files + n_subframe_chain_files - 1) // max(1, n_subframe_chain_files))
        
        file_stats = generate_calibration_file(
            filepath,
            seed=seed,
            file_index=i,
            n_runs=runs_per_calib * 2  # Overlap coverage
        )
        
        if file_stats:
            stats['subframe_files'].append(file_stats)
            stats['total_size_mb'] += file_stats['size_mb']
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"CHAIN GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Main data files: {len(stats['main_files'])}")
    print(f"  Subframe chain files: {len(stats['subframe_files'])}")
    print(f"  Total rows: {stats['total_rows']:,}")
    print(f"  Total size: {stats['total_size_mb']:.1f} MB")
    print()
    print(f"Main data pattern: {output_dir}/data_run*.root:tree")
    if stats['subframe_files']:
        print(f"Subframe pattern:  {output_dir}/calib_*.root:tree")
    print()
    print("Known relationships for invariance testing:")
    print("  - y_derived = 2 * x (exact, no noise)")
    print("  - gain[sec] = 1.0 + 0.01 * sec (in SectorCalib)")
    print("  - gain[run,sec] = 1.0 + 0.01*sec + 0.001*(run-1000) (in calib chain)")
    print(f"{'='*60}\n")
    
    return stats


def verify_file(filepath, rdf_mode=False):
    """Verify the generated ROOT file can be read.
    
    Parameters
    ----------
    filepath : str
        Path to the ROOT file to verify
    rdf_mode : bool
        If True, expect RDF-style file with aliases and 4 subframes.
        If False, expect standard file with y_derived invariant.
    """
    try:
        print(f"\nVerifying file: {filepath}")
        
        # Test with AliasDataFrame
        from AliasDataFrame import AliasDataFrame
        
        adf = AliasDataFrame.read_tree(filepath, 'tree', load_subframes=True)
        print(f"  ✓ AliasDataFrame loaded: {len(adf.df):,} rows")
        print(f"    Columns: {list(adf.df.columns)[:8]}...")
        
        subframes = list(adf._subframes.subframes.keys()) if hasattr(adf, '_subframes') else []
        print(f"    Subframes: {subframes}")
        
        aliases = list(adf.aliases.keys()) if hasattr(adf, 'aliases') else []
        if aliases:
            print(f"    Aliases ({len(aliases)}): {aliases[:5]}...")
        elif rdf_mode:
            print(f"  ⚠ Warning: No aliases found in RDF mode file")
        
        # Check for known relationships (standard mode)
        if not rdf_mode and 'x' in adf.df.columns and 'y_derived' in adf.df.columns:
            expected = 2.0 * adf.df['x'].values
            actual = adf.df['y_derived'].values
            if np.allclose(expected, actual):
                print(f"  ✓ Invariant verified: y_derived = 2 * x")
            else:
                print(f"  ✗ Invariant FAILED: y_derived != 2 * x")
                return False
        
        return True
            
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_chain(output_dir, n_files):
    """Verify chain files can be read together."""
    try:
        print(f"\nVerifying chain in: {output_dir}")
        
        from AliasDataFrame import AliasDataFrame
        
        pattern = os.path.join(output_dir, 'data_run*.root:tree')
        adf = AliasDataFrame.read_chain_lazy(pattern)
        
        print(f"  ✓ Chain reader created")
        print(f"    Files: {len(adf._lazy_reader._files)}")
        print(f"    Available branches: {adf._lazy_reader.available_branches[:5]}...")
        
        # Load some data to verify
        adf.load_branches(['x', 'y_derived', 'file_idx'])
        print(f"  ✓ Loaded {len(adf.df):,} total rows from chain")
        
        # Verify invariant across chain
        expected = 2.0 * adf.df['x'].values
        actual = adf.df['y_derived'].values
        if np.allclose(expected, actual):
            print(f"  ✓ Invariant verified across chain: y_derived = 2 * x")
        else:
            print(f"  ✗ Invariant FAILED across chain")
            return False
        
        # Verify file indices are present
        unique_files = np.unique(adf.df['file_idx'].values)
        print(f"  ✓ File indices present: {list(unique_files)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Chain verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic ROOT files for benchmarks and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file (default)
    python generate_synthetic_data.py
    python generate_synthetic_data.py --output test_data.root
    python generate_synthetic_data.py --rows 500000 --tracks 50000
    
    # Chain mode (Phase 6.8)
    python generate_synthetic_data.py --chain 5 -o chain_data/
    python generate_synthetic_data.py --chain 5 --subframe-chain 3 -o chain_data/
    
    # RDF Benchmark Mode
    python generate_synthetic_data.py --rdf
    python generate_synthetic_data.py --rdf --rows 1000000
    python generate_synthetic_data.py --rdf --sparse  # Test sparse key algorithm
    
    # Verify generated files
    python generate_synthetic_data.py --verify
    python generate_synthetic_data.py --chain 3 -o chain_data/ --verify

Known Relationships (for invariance testing):
    - y_derived = 2 * x (exact, no noise)
    - gain[sec] = 1.0 + 0.01 * sec (SectorCalib subframe)
    - gain[run,sec] = 1.0 + 0.01*sec + 0.001*(run-1000) (calibration chain)
        """
    )
    
    # Output options
    parser.add_argument('--output', '-o', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 'synthetic_data.root'),
                        help='Output file path or directory (default: benchmarks/synthetic_data.root)')
    
    # Size options
    parser.add_argument('--rows', type=int, default=DEFAULT_ROWS,
                        help=f'Number of rows in main tree (default: {DEFAULT_ROWS:,})')
    parser.add_argument('--tracks', type=int, default=DEFAULT_TRACKS,
                        help=f'Number of tracks in subframe (default: {DEFAULT_TRACKS:,}, ignored in --rdf mode)')
    
    # Chain options (Phase 6.8)
    parser.add_argument('--chain', type=int, default=0,
                        help='Generate N files for chain testing (creates data_run001.root, etc.)')
    parser.add_argument('--subframe-chain', type=int, default=0,
                        help='Generate N calibration files for subframe chain testing')
    
    # RDF options (Phase 3/5 - backward compatible)
    parser.add_argument('--rdf', action='store_true',
                        help='Generate RDF benchmark data (4 subframes, multi-key indices)')
    parser.add_argument('--sparse', action='store_true',
                        help='Use sparse (non-contiguous) key values (only with --rdf)')
    
    # Other options
    parser.add_argument('--seed', type=int, default=RNG_SEED,
                        help=f'Random seed (default: {RNG_SEED})')
    parser.add_argument('--verify', action='store_true',
                        help='Verify file(s) after generation')
    
    args = parser.parse_args()
    
    # RDF mode (Phase 3/5 benchmarks - backward compatible)
    if args.rdf:
        stats = generate_rdf_synthetic_root(
            output_path=args.output,
            n_rows=args.rows,
            seed=args.seed,
            sparse_keys=args.sparse
        )
        
        if stats is None:
            sys.exit(1)
        
        if args.verify:
            if not verify_file(args.output, rdf_mode=True):
                sys.exit(1)
    
    # Chain mode (Phase 6.8)
    elif args.chain > 0:
        stats = generate_chain_data(
            output_dir=args.output,
            n_chain_files=args.chain,
            n_subframe_chain_files=args.subframe_chain,
            rows_per_file=args.rows,
            tracks_per_file=args.tracks,
            seed=args.seed
        )
        
        if args.verify and stats:
            if not verify_chain(args.output, args.chain):
                sys.exit(1)
    
    # Single file mode (default)
    else:
        stats = generate_synthetic_root(
            output_path=args.output,
            n_rows=args.rows,
            n_tracks=args.tracks,
            seed=args.seed,
            file_index=0,
            include_derived=True
        )
        
        if stats is None:
            sys.exit(1)
        
        if args.verify:
            if not verify_file(args.output, rdf_mode=False):
                sys.exit(1)
    
    print(f"\nDone. File ready at: {args.output}")


if __name__ == '__main__':
    main()
