"""
Test cases for dependency_tree() and describe_aliases(expr_width=) features.
"""

import numpy as np
import pandas as pd
import pytest
import io
import sys


def test_dependency_tree_basic():
    """Test basic dependency tree output."""
    from AliasDataFrame import AliasDataFrame
    
    df = pd.DataFrame({
        'x': np.random.randn(100).astype(np.float32),
        'y': np.random.randn(100).astype(np.float32),
    })
    
    adf = AliasDataFrame(df)
    adf.add_alias('r', 'sqrt(x**2 + y**2)')
    adf.add_alias('r_normalized', 'r / 10')
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    adf.dependency_tree('r_normalized')
    
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Verify structure
    assert 'r_normalized' in output
    assert 'r = sqrt(x**2 + y**2)' in output
    assert 'x [column]' in output
    assert 'y [column]' in output
    print("✅ test_dependency_tree_basic PASSED")


def test_dependency_tree_max_depth():
    """Test dependency tree with max_depth limit."""
    from AliasDataFrame import AliasDataFrame
    
    df = pd.DataFrame({
        'a': np.random.randn(100).astype(np.float32),
    })
    
    adf = AliasDataFrame(df)
    adf.add_alias('level1', 'a + 1')
    adf.add_alias('level2', 'level1 + 1')
    adf.add_alias('level3', 'level2 + 1')
    
    # With max_depth=1, should only show level2, not level1
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    adf.dependency_tree('level3', max_depth=1)
    
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    assert 'level3' in output
    assert 'level2' in output
    # level1 should NOT appear since we limited to depth 1
    lines = output.strip().split('\n')
    # Should have root + 1 level of children
    assert len(lines) == 2  # level3 + level2
    print("✅ test_dependency_tree_max_depth PASSED")


def test_dependency_tree_show_expr_false():
    """Test dependency tree with show_expr=False."""
    from AliasDataFrame import AliasDataFrame
    
    df = pd.DataFrame({
        'x': np.random.randn(100).astype(np.float32),
    })
    
    adf = AliasDataFrame(df)
    adf.add_alias('result', 'x + 1')
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    adf.dependency_tree('result', show_expr=False)
    
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Should NOT contain '=' for expression
    assert 'result' in output
    assert '= x + 1' not in output
    print("✅ test_dependency_tree_show_expr_false PASSED")


def test_dependency_tree_with_subframe():
    """Test dependency tree with subframe references."""
    from AliasDataFrame import AliasDataFrame
    
    df = pd.DataFrame({
        'track_id': np.arange(100, dtype=np.int32),
        'x': np.random.randn(100).astype(np.float32),
    })
    
    track_df = pd.DataFrame({
        'track_id': np.arange(20, dtype=np.int32),
        'mass': np.random.randn(20).astype(np.float32),
    })
    
    adf = AliasDataFrame(df)
    track_adf = AliasDataFrame(track_df)
    adf.register_subframe('T', track_adf, 'track_id')
    
    adf.add_alias('result', 'x + T.mass')
    
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    adf.dependency_tree('result')
    
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    assert 'result' in output
    assert 'T.mass [subframe]' in output
    assert 'x [column]' in output
    print("✅ test_dependency_tree_with_subframe PASSED")


def test_describe_aliases_expr_width():
    """Test describe_aliases with expr_width parameter."""
    from AliasDataFrame import AliasDataFrame
    
    df = pd.DataFrame({
        'x': np.random.randn(100).astype(np.float32),
    })
    
    adf = AliasDataFrame(df)
    # Add an alias with a long expression
    long_expr = 'x + ' * 20 + 'x'  # ~80 chars
    adf.add_alias('long_alias', long_expr)
    
    # Test with narrow width
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    adf.describe_aliases(expr_width=30, names=['long_alias'])
    
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Should be truncated
    assert '...' in output
    print("✅ test_describe_aliases_expr_width PASSED")


def test_describe_aliases_expr_width_none():
    """Test describe_aliases with expr_width=None (no truncation)."""
    from AliasDataFrame import AliasDataFrame
    
    df = pd.DataFrame({
        'x': np.random.randn(100).astype(np.float32),
    })
    
    adf = AliasDataFrame(df)
    long_expr = 'x + ' * 20 + 'x'  # ~80 chars
    adf.add_alias('long_alias', long_expr)
    
    # Test with no truncation
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    adf.describe_aliases(expr_width=None, names=['long_alias'])
    
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Should NOT be truncated
    assert long_expr in output
    assert '...' not in output.split('\n')[3]  # Expression line should not have ...
    print("✅ test_describe_aliases_expr_width_none PASSED")


if __name__ == '__main__':
    test_dependency_tree_basic()
    test_dependency_tree_max_depth()
    test_dependency_tree_show_expr_false()
    test_dependency_tree_with_subframe()
    test_describe_aliases_expr_width()
    test_describe_aliases_expr_width_none()
    print("\n✅ All dependency_tree and expr_width tests PASSED!")
