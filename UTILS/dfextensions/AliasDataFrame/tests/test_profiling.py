"""
Test cases for profiling functionality with explicit output parameters.
"""

import numpy as np
import pandas as pd
import pytest
import pstats
import sys
import io
from pathlib import Path


class TestProfiling:
    """Test profiling with explicit profile_text and profile_binary parameters."""
    
    @pytest.fixture
    def adf(self):
        """Create a simple AliasDataFrame for testing."""
        from AliasDataFrame import AliasDataFrame
        
        df = pd.DataFrame({
            'x': np.random.randn(1000).astype(np.float32),
            'y': np.random.randn(1000).astype(np.float32),
        })
        adf = AliasDataFrame(df)
        adf.add_alias('r', 'sqrt(x**2 + y**2)')
        adf.add_alias('result', 'r * 2')
        return adf
    
    def test_profile_stdout_only(self, adf, capsys):
        """Test profile=True prints to stdout without creating files."""
        adf.materialize_aliases(names=['result'], profile=True)
        
        captured = capsys.readouterr()
        # Should contain profiling output
        assert 'cumulative' in captured.out.lower() or 'tottime' in captured.out.lower()
    
    def test_profile_text_output(self, adf, tmp_path):
        """Test text profile output."""
        text_file = tmp_path / "profile.txt"
        adf.materialize_aliases(names=['result'], profile_text=str(text_file))
        
        assert text_file.exists()
        content = text_file.read_text()
        assert 'cumulative' in content.lower() or 'tottime' in content.lower()
    
    def test_profile_binary_output(self, adf, tmp_path):
        """Test binary profile output."""
        prof_file = tmp_path / "profile.prof"
        adf.materialize_aliases(names=['result'], profile_binary=str(prof_file))
        
        assert prof_file.exists()
        # Verify it's valid binary profile
        stats = pstats.Stats(str(prof_file))
        assert stats.total_calls > 0
    
    def test_profile_both_outputs(self, adf, tmp_path, capsys):
        """Test both text and binary output together."""
        text_file = tmp_path / "profile.txt"
        prof_file = tmp_path / "profile.prof"
        
        adf.materialize_aliases(
            names=['result'], 
            profile=True,  # Also print to stdout
            profile_text=str(text_file),
            profile_binary=str(prof_file)
        )
        
        # Both files should exist
        assert text_file.exists()
        assert prof_file.exists()
        
        # stdout should have profiling output
        captured = capsys.readouterr()
        assert 'cumulative' in captured.out.lower() or 'tottime' in captured.out.lower()
        
        # Text file should have content
        content = text_file.read_text()
        assert len(content) > 100
        
        # Binary file should be valid
        stats = pstats.Stats(str(prof_file))
        assert stats.total_calls > 0
    
    def test_profile_text_only_no_stdout(self, adf, tmp_path, capsys):
        """Test profile_text without profile=True doesn't print full profile to stdout."""
        text_file = tmp_path / "profile.txt"
        adf.materialize_aliases(names=['result'], profile_text=str(text_file))
        
        captured = capsys.readouterr()
        # Should only see the "saved to" message, not the full profile
        assert '[profiler] Text saved to:' in captured.out
        # The full cumulative table should NOT be in stdout (only in file)
        lines = captured.out.strip().split('\n')
        # Should be just a few lines (save messages), not 80+ lines of profile
        assert len(lines) < 10
    
    def test_profile_binary_only_no_stdout(self, adf, tmp_path, capsys):
        """Test profile_binary without profile=True doesn't print to stdout."""
        prof_file = tmp_path / "profile.prof"
        adf.materialize_aliases(names=['result'], profile_binary=str(prof_file))
        
        captured = capsys.readouterr()
        # Should only see the "saved to" message
        assert '[profiler] Binary saved to:' in captured.out
        lines = captured.out.strip().split('\n')
        assert len(lines) < 5
    
    def test_no_profiling_no_output(self, adf, tmp_path, capsys):
        """Test that without any profile params, no profiling happens."""
        adf.materialize_aliases(names=['result'])
        
        captured = capsys.readouterr()
        # Should be no profiler output
        assert '[profiler]' not in captured.out
        assert 'cumulative' not in captured.out.lower()
    
    def test_materialize_alias_singular_profiling(self, adf, tmp_path):
        """Test profiling also works with materialize_alias (singular)."""
        from AliasDataFrame import AliasDataFrame
        
        df = pd.DataFrame({
            'x': np.random.randn(1000).astype(np.float32),
        })
        adf2 = AliasDataFrame(df)
        adf2.add_alias('doubled', 'x * 2')
        
        text_file = tmp_path / "single.txt"
        prof_file = tmp_path / "single.prof"
        
        adf2.materialize_alias('doubled', profile_text=str(text_file), profile_binary=str(prof_file))
        
        assert text_file.exists()
        assert prof_file.exists()


def test_profile_text_output():
    """Standalone test for text profile output."""
    from AliasDataFrame import AliasDataFrame
    import tempfile
    
    df = pd.DataFrame({
        'x': np.random.randn(100).astype(np.float32),
        'y': np.random.randn(100).astype(np.float32),
    })
    adf = AliasDataFrame(df)
    adf.add_alias('result', 'x + y')
    
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        text_path = f.name
    
    try:
        adf.materialize_aliases(names=['result'], profile_text=text_path)
        content = Path(text_path).read_text()
        assert 'cumulative' in content.lower() or 'tottime' in content.lower()
        print("✅ test_profile_text_output PASSED")
    finally:
        Path(text_path).unlink(missing_ok=True)


def test_profile_binary_output():
    """Standalone test for binary profile output."""
    from AliasDataFrame import AliasDataFrame
    import tempfile
    
    df = pd.DataFrame({
        'x': np.random.randn(100).astype(np.float32),
        'y': np.random.randn(100).astype(np.float32),
    })
    adf = AliasDataFrame(df)
    adf.add_alias('result', 'x + y')
    
    with tempfile.NamedTemporaryFile(suffix='.prof', delete=False) as f:
        prof_path = f.name
    
    try:
        adf.materialize_aliases(names=['result'], profile_binary=prof_path)
        stats = pstats.Stats(prof_path)
        assert stats.total_calls > 0
        print("✅ test_profile_binary_output PASSED")
    finally:
        Path(prof_path).unlink(missing_ok=True)


def test_profile_both_outputs():
    """Standalone test for both text and binary output."""
    from AliasDataFrame import AliasDataFrame
    import tempfile
    
    df = pd.DataFrame({
        'x': np.random.randn(100).astype(np.float32),
        'y': np.random.randn(100).astype(np.float32),
    })
    adf = AliasDataFrame(df)
    adf.add_alias('result', 'x + y')
    
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
        text_path = f.name
    with tempfile.NamedTemporaryFile(suffix='.prof', delete=False) as f:
        prof_path = f.name
    
    try:
        adf.materialize_aliases(
            names=['result'], 
            profile=True,
            profile_text=text_path,
            profile_binary=prof_path
        )
        
        # Verify text file
        content = Path(text_path).read_text()
        assert len(content) > 100
        
        # Verify binary file
        stats = pstats.Stats(prof_path)
        assert stats.total_calls > 0
        
        print("✅ test_profile_both_outputs PASSED")
    finally:
        Path(text_path).unlink(missing_ok=True)
        Path(prof_path).unlink(missing_ok=True)


if __name__ == '__main__':
    test_profile_text_output()
    test_profile_binary_output()
    test_profile_both_outputs()
    print("\n✅ All profiling tests PASSED!")
