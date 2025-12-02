"""
Test case for cleanTemporary bug fix.

Bug: cleanTemporary=True was not removing intermediate subframe columns
after materialize_aliases().

Fix: Added cleanup of columns matching pattern {col}__{subframe_name}
"""

import numpy as np
import pandas as pd
import pytest


def test_clean_temporary_subframe_columns():
    """Verify cleanTemporary removes intermediate subframe columns."""
    # Import here to avoid module-level issues
    from AliasDataFrame import AliasDataFrame
    
    # Create main DataFrame
    n = 1000
    np.random.seed(42)
    main_df = pd.DataFrame({
        'track_id': np.random.randint(0, 100, n, dtype=np.int32),
        'x': np.random.randn(n).astype(np.float32),
        'y': np.random.randn(n).astype(np.float32),
    })
    
    # Create subframe
    n_tracks = 100
    track_df = pd.DataFrame({
        'track_id': np.arange(n_tracks, dtype=np.int32),
        'mass': np.random.randn(n_tracks).astype(np.float32) + 1.0,
        'charge': np.random.choice([-1, 1], n_tracks).astype(np.int8),
    })
    
    adf = AliasDataFrame(main_df)
    track_adf = AliasDataFrame(track_df)
    adf.register_subframe('T', track_adf, 'track_id')
    
    # Define aliases with subframe dependencies
    adf.add_alias('r', 'sqrt(x**2 + y**2)')  # Simple alias (intermediate)
    adf.add_alias('track_mass', 'T.mass')    # Subframe ref (intermediate)
    adf.add_alias('final_result', 'r + track_mass')  # Final target
    
    # Record original columns
    original_columns = set(adf.df.columns)
    
    # Materialize with cleanup
    adf.materialize_aliases(
        names=['final_result'],
        with_dependencies=True,
        cleanTemporary=True
    )
    
    final_columns = set(adf.df.columns)
    
    # Check: only final_result should be added
    new_columns = final_columns - original_columns
    assert new_columns == {'final_result'}, \
        f"Expected only 'final_result', got: {new_columns}"
    
    # Check: no subframe intermediate columns should remain
    subframe_temps = [c for c in adf.df.columns if '__' in c]
    assert len(subframe_temps) == 0, \
        f"Temporary subframe columns not cleaned: {subframe_temps}"
    
    # Check: intermediate alias 'r' should not remain
    assert 'r' not in adf.df.columns, "'r' intermediate should have been cleaned"
    
    # Check: intermediate alias 'track_mass' should not remain  
    assert 'track_mass' not in adf.df.columns, "'track_mass' intermediate should have been cleaned"
    
    print("✅ test_clean_temporary_subframe_columns PASSED")


def test_clean_temporary_preserves_targets():
    """Verify cleanTemporary does NOT remove explicitly requested aliases."""
    from AliasDataFrame import AliasDataFrame
    
    n = 1000
    np.random.seed(42)
    main_df = pd.DataFrame({
        'track_id': np.random.randint(0, 100, n, dtype=np.int32),
        'x': np.random.randn(n).astype(np.float32),
    })
    
    track_df = pd.DataFrame({
        'track_id': np.arange(100, dtype=np.int32),
        'mass': np.random.randn(100).astype(np.float32),
    })
    
    adf = AliasDataFrame(main_df)
    track_adf = AliasDataFrame(track_df)
    adf.register_subframe('T', track_adf, 'track_id')
    
    adf.add_alias('track_mass', 'T.mass')
    adf.add_alias('result', 'x + track_mass')
    
    original_columns = set(adf.df.columns)
    
    # Request BOTH aliases as targets
    adf.materialize_aliases(
        names=['track_mass', 'result'],
        with_dependencies=True,
        cleanTemporary=True
    )
    
    final_columns = set(adf.df.columns)
    new_columns = final_columns - original_columns
    
    # Both should be preserved since both were requested
    assert 'track_mass' in new_columns, "Requested 'track_mass' should be preserved"
    assert 'result' in new_columns, "Requested 'result' should be preserved"
    
    # But subframe join column should be cleaned
    subframe_temps = [c for c in adf.df.columns if '__T' in c]
    assert len(subframe_temps) == 0, \
        f"Temporary subframe columns not cleaned: {subframe_temps}"
    
    print("✅ test_clean_temporary_preserves_targets PASSED")


def test_clean_temporary_multiple_subframes():
    """Verify cleanTemporary works with multiple subframes."""
    from AliasDataFrame import AliasDataFrame
    
    n = 500
    np.random.seed(42)
    main_df = pd.DataFrame({
        'track_id': np.random.randint(0, 50, n, dtype=np.int32),
        'cluster_id': np.random.randint(0, 30, n, dtype=np.int32),
        'x': np.random.randn(n).astype(np.float32),
    })
    
    track_df = pd.DataFrame({
        'track_id': np.arange(50, dtype=np.int32),
        'pt': np.random.randn(50).astype(np.float32),
    })
    
    cluster_df = pd.DataFrame({
        'cluster_id': np.arange(30, dtype=np.int32),
        'energy': np.random.randn(30).astype(np.float32),
    })
    
    adf = AliasDataFrame(main_df)
    adf.register_subframe('T', AliasDataFrame(track_df), 'track_id')
    adf.register_subframe('C', AliasDataFrame(cluster_df), 'cluster_id')
    
    adf.add_alias('combined', 'T.pt + C.energy + x')
    
    original_columns = set(adf.df.columns)
    
    adf.materialize_aliases(
        names=['combined'],
        with_dependencies=True,
        cleanTemporary=True
    )
    
    final_columns = set(adf.df.columns)
    new_columns = final_columns - original_columns
    
    # Only 'combined' should be added
    assert new_columns == {'combined'}, f"Expected only 'combined', got: {new_columns}"
    
    # No subframe columns from either T or C
    subframe_temps = [c for c in adf.df.columns if '__T' in c or '__C' in c]
    assert len(subframe_temps) == 0, \
        f"Temporary columns not cleaned: {subframe_temps}"
    
    print("✅ test_clean_temporary_multiple_subframes PASSED")


def test_no_cleanup_when_disabled():
    """Verify subframe columns are preserved when cleanTemporary=False."""
    from AliasDataFrame import AliasDataFrame
    
    n = 500
    np.random.seed(42)
    main_df = pd.DataFrame({
        'track_id': np.random.randint(0, 50, n, dtype=np.int32),
        'x': np.random.randn(n).astype(np.float32),
    })
    
    track_df = pd.DataFrame({
        'track_id': np.arange(50, dtype=np.int32),
        'mass': np.random.randn(50).astype(np.float32),
    })
    
    adf = AliasDataFrame(main_df)
    adf.register_subframe('T', AliasDataFrame(track_df), 'track_id')
    
    adf.add_alias('result', 'x + T.mass')
    
    adf.materialize_aliases(
        names=['result'],
        with_dependencies=True,
        cleanTemporary=False  # Disable cleanup
    )
    
    # Subframe column SHOULD remain when cleanup is disabled
    subframe_temps = [c for c in adf.df.columns if '__T' in c]
    assert len(subframe_temps) > 0, \
        "Subframe columns should be preserved when cleanTemporary=False"
    
    print("✅ test_no_cleanup_when_disabled PASSED")


if __name__ == '__main__':
    test_clean_temporary_subframe_columns()
    test_clean_temporary_preserves_targets()
    test_clean_temporary_multiple_subframes()
    test_no_cleanup_when_disabled()
    print("\n✅ All cleanTemporary tests PASSED!")
