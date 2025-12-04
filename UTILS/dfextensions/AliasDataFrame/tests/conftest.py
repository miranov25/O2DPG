"""
Pytest fixtures for AliasDataFrameRDF tests.

Provides session-scoped test data with all 4 subframes and proper indices.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

# Add parent directory to path
_this_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_this_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)


def create_rdf_test_data(filepath: str):
    """
    Create test data with all 4 subframes and proper indices.
    Mirrors real calibration structure.
    
    Structure:
    - Main tree: 10,000 rows
    - Subframe T: 1-key index (track_tf_uid), 100 entries
    - Subframe R: 1-key index (firstTForbit), 50 entries
    - Subframe DTrack0: 3-key index (side, row, drift25), 8512 entries
    - Subframe DITS0FitSide: 2-key index (drift25, side), 56 entries
    
    Parameters
    ----------
    filepath : str
        Output ROOT file path
        
    Returns
    -------
    tuple
        (filepath, aDF) - path to file and the AliasDataFrame object with aliases
    """
    from AliasDataFrame import AliasDataFrame
    from itertools import product
    
    np.random.seed(42)  # Reproducible
    n_rows = 10_000
    
    # Main tree columns
    main_df = pd.DataFrame({
        'track_tf_uid': np.random.randint(0, 100, n_rows),  # For T join (1-key)
        'firstTForbit': np.random.randint(0, 50, n_rows),   # For R join (1-key)
        'side': np.random.randint(0, 2, n_rows),            # For DTrack0 (3-key)
        'row': np.random.randint(0, 152, n_rows),           # For DTrack0 (3-key)
        'drift25': np.random.randint(0, 28, n_rows),        # For DTrack0 (3-key)
        'mX': np.random.randn(n_rows).astype(np.float32),
        'mY': np.random.randn(n_rows).astype(np.float32),
        'mZ': np.random.randn(n_rows).astype(np.float32),
        'x': np.random.randn(n_rows).astype(np.float32),
        'y': np.random.randn(n_rows).astype(np.float32),
    })
    
    # Subframe T: 1-key index (track_tf_uid)
    t_df = pd.DataFrame({
        'track_tf_uid': np.arange(100),
        'mP2': np.random.randn(100).astype(np.float32),
        'mP3': np.random.randn(100).astype(np.float32),
        'mP4': np.random.randn(100).astype(np.float32),
        'dy': np.random.randn(100).astype(np.float32) * 0.1,
        'dz': np.random.randn(100).astype(np.float32) * 0.1,
    })
    
    # Subframe R: 1-key index (firstTForbit)
    r_df = pd.DataFrame({
        'firstTForbit': np.arange(50),
        'refX': np.random.randn(50).astype(np.float32),
    })
    
    # Subframe DTrack0: 3-key index (side, row, drift25)
    # Create all combinations: 2 * 152 * 28 = 8512 entries
    keys = list(product(range(2), range(152), range(28)))
    dtrack_df = pd.DataFrame({
        'side': [k[0] for k in keys],
        'row': [k[1] for k in keys],
        'drift25': [k[2] for k in keys],
        'dyC2_median': np.random.randn(len(keys)).astype(np.float32) * 0.01,
        'dzC2_median': np.random.randn(len(keys)).astype(np.float32) * 0.01,
    })
    # Create composite key for N>2 key join (same algorithm as AliasDataFrameTree.C)
    # __adf_key__ = k0 + k1*max0 + k2*max0*max1
    max_side, max_row, max_drift = 2, 152, 28
    dtrack_df['__adf_key_DTrack0__'] = (
        dtrack_df['side'] + 
        dtrack_df['row'] * max_side + 
        dtrack_df['drift25'] * max_side * max_row
    ).astype(np.int64)
    
    # Also add composite key to main tree for join
    main_df['__adf_key_DTrack0__'] = (
        main_df['side'] + 
        main_df['row'] * max_side + 
        main_df['drift25'] * max_side * max_row
    ).astype(np.int64)
    
    # Subframe DITS0FitSide: 2-key index (drift25, side)
    keys2 = list(product(range(28), range(2)))  # 28 * 2 = 56 entries
    dits_df = pd.DataFrame({
        'drift25': [k[0] for k in keys2],
        'side': [k[1] for k in keys2],
        'itsParam': np.random.randn(len(keys2)).astype(np.float32),
    })
    
    # Create AliasDataFrame
    aDF = AliasDataFrame(main_df)
    
    # Register subframes with proper index columns
    aDF.register_subframe('T', AliasDataFrame(t_df), index_columns='track_tf_uid')
    aDF.register_subframe('R', AliasDataFrame(r_df), index_columns='firstTForbit')
    # For 3-key subframe, use composite key (matches AliasDataFrameTree.C behavior)
    aDF.register_subframe('DTrack0', AliasDataFrame(dtrack_df), 
                          index_columns='__adf_key_DTrack0__')
    aDF.register_subframe('DITS0FitSide', AliasDataFrame(dits_df), 
                          index_columns=['drift25', 'side'])
    
    # Add test aliases (representative subset)
    # These cover various patterns: subframe access, arithmetic, boolean
    # Note: Use C++-compatible function names (tan, abs) not numpy (np.tan, np.abs)
    aDF.add_alias('z_calc', 'tan(T.mP3) * drift25')
    aDF.add_alias('dy_c', 'T.mP2 - mY')
    aDF.add_alias('dz_c', 'T.mP4 - mZ')
    aDF.add_alias('dyC2', 'dy_c - DTrack0.dyC2_median')
    aDF.add_alias('dzC2', 'dz_c - DTrack0.dzC2_median')
    aDF.add_alias('isValid', '(row < 152) & (abs(dyC2) < 2)')
    
    # Export with composite indices
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    aDF.export_tree(filepath, "tree")
    
    print(f"Created test data: {filepath}")
    print(f"  Main tree: {n_rows} rows")
    print(f"  Subframe T: {len(t_df)} entries (1-key)")
    print(f"  Subframe R: {len(r_df)} entries (1-key)")
    print(f"  Subframe DTrack0: {len(dtrack_df)} entries (3-key)")
    print(f"  Subframe DITS0FitSide: {len(dits_df)} entries (2-key)")
    print(f"  Aliases: {len(aDF.aliases)} defined")  # Use .aliases property
    
    # Return both filepath and aDF object (with aliases)
    return filepath, aDF


@pytest.fixture(scope="session")
def rdf_test_data(tmp_path_factory):
    """
    Session-scoped fixture that creates test data once per test session.
    
    Returns tuple of (filepath, aDF) where aDF has the aliases defined.
    """
    filepath = tmp_path_factory.mktemp("data") / "rdf_test_data.root"
    return create_rdf_test_data(str(filepath))


@pytest.fixture(scope="session")
def rdf_test_file(rdf_test_data):
    """Returns path to ROOT file with test data."""
    return rdf_test_data[0]


@pytest.fixture(scope="session")
def rdf_test_adf(rdf_test_data):
    """Returns AliasDataFrame with test aliases defined."""
    return rdf_test_data[1]


# =============================================================================
# Persistent Fixture Data (optional - for reuse across test runs)
# =============================================================================

# Path to persistent fixture data (relative to tests/ directory)
PERSISTENT_FIXTURE_PATH = os.path.join(_this_dir, "fixtures", "rdf_test_data.root")


def get_or_create_persistent_fixture():
    """
    Get or create persistent fixture data.
    
    If fixtures/rdf_test_data.root exists, return it.
    Otherwise create it.
    
    This allows reusing the same test data across multiple test runs,
    which is faster than recreating it each time.
    
    Usage:
        # In conftest.py, replace rdf_test_data fixture with:
        @pytest.fixture(scope="session")
        def rdf_test_data():
            return get_or_create_persistent_fixture()
    """
    if os.path.exists(PERSISTENT_FIXTURE_PATH):
        print(f"Using existing fixture: {PERSISTENT_FIXTURE_PATH}")
        return _recreate_adf_with_schema(PERSISTENT_FIXTURE_PATH)
    else:
        print(f"Creating new fixture: {PERSISTENT_FIXTURE_PATH}")
        return create_rdf_test_data(PERSISTENT_FIXTURE_PATH)


def _recreate_adf_with_schema(filepath):
    """
    Recreate AliasDataFrame with schema from existing file.
    
    Since aliases aren't stored in the ROOT file, we recreate them here.
    """
    from AliasDataFrame import AliasDataFrame
    import uproot
    
    # Load the main DataFrame
    with uproot.open(filepath) as f:
        tree = f["tree"]
        main_df = tree.arrays(library="pd")
    
    aDF = AliasDataFrame(main_df)
    
    # Add the same aliases (must match create_rdf_test_data)
    aDF.add_alias('z_calc', 'tan(T.mP3) * drift25')
    aDF.add_alias('dy_c', 'T.mP2 - mY')
    aDF.add_alias('dz_c', 'T.mP4 - mZ')
    aDF.add_alias('dyC2', 'dy_c - DTrack0.dyC2_median')
    aDF.add_alias('dzC2', 'dz_c - DTrack0.dzC2_median')
    aDF.add_alias('isValid', '(row < 152) & (abs(dyC2) < 2)')
    
    # Add subframe info to schema (for setup_tree_with_friends)
    # Schema uses 'index' key, not 'index_columns'
    # DTrack0 uses composite key for 3-key join
    aDF._schema['subframes'] = {
        'T': {'index': ['track_tf_uid']},
        'R': {'index': ['firstTForbit']},
        'DTrack0': {'index': ['__adf_key_DTrack0__']},  # Composite key
        'DITS0FitSide': {'index': ['drift25', 'side']},
    }
    
    return filepath, aDF


# Allow running this file directly to create persistent fixture data
if __name__ == '__main__':
    os.makedirs(os.path.dirname(PERSISTENT_FIXTURE_PATH), exist_ok=True)
    filepath, aDF = create_rdf_test_data(PERSISTENT_FIXTURE_PATH)
    print(f"\nPersistent fixture created at: {filepath}")
    print(f"This file can be reused across test runs.")
