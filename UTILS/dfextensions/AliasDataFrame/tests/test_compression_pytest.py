"""
Test compression monitoring functionality.
"""

import pytest
import pandas as pd
import numpy as np
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_with_compression():
    """Create AliasDataFrame with various compression configurations."""
    df = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100) * 10,
        'z': np.random.uniform(0, 100, 100),
    })
    adf = AliasDataFrame(df)
    
    # Compression without monitor
    adf.compression_info['no_monitor'] = {
        'state': 'compressed',
        'compressed_col': 'no_monitor_c',
        'compressed_dtype': 'int16',
        'decompressed_dtype': 'float32',
        'compress_expr': 'round(x*100)',
        'decompress_expr': 'x_c/100.',
        'precision': {
            'rmse': 0.005,
            'max_error': 0.01,
            'mean_error': 0.0,
            'fraction_nonfinite': 0.0,
        }
    }
    
    # Absolute monitor - will pass
    adf.compression_info['abs_pass'] = {
        'state': 'compressed',
        'compressed_col': 'abs_pass_c',
        'compressed_dtype': 'int16',
        'decompressed_dtype': 'float32',
        'compress_expr': 'round(y*10)',
        'decompress_expr': 'y_c/10.',
        'precision': {
            'rmse': 0.05,
            'max_error': 0.1,
            'mean_error': 0.0,
            'fraction_nonfinite': 0.0,
        },
        'monitor': {
            'type': 'absolute',
            'threshold': 0.1
        }
    }
    
    # Absolute monitor - will fail
    adf.compression_info['abs_fail'] = {
        'state': 'compressed',
        'compressed_col': 'abs_fail_c',
        'compressed_dtype': 'int8',
        'decompressed_dtype': 'float32',
        'compress_expr': 'round(z)',
        'decompress_expr': 'float(z_c)',
        'precision': {
            'rmse': 0.5,
            'max_error': 1.0,
            'mean_error': 0.0,
            'fraction_nonfinite': 0.0,
        },
        'monitor': {
            'type': 'absolute',
            'threshold': 0.1
        }
    }
    
    # Relative monitor - will pass
    adf.compression_info['rel_pass'] = {
        'state': 'decompressed',
        'compressed_col': 'rel_pass_c',
        'compressed_dtype': 'int16',
        'decompressed_dtype': 'float32',
        'compress_expr': 'round(z*10)',
        'decompress_expr': 'z_c/10.',
        'precision': {
            'rmse': 0.05,
            'max_error': 0.1,
            'mean_error': 0.0,
            'fraction_nonfinite': 0.0,
            'data_range': (0, 100),
        },
        'monitor': {
            'type': 'relative',
            'relative_to': 'data_range',
            'threshold': 0.01
        }
    }
    
    # Function monitor - will fail
    adf.compression_info['func_fail'] = {
        'state': 'compressed',
        'compressed_col': 'func_fail_c',
        'compressed_dtype': 'uint8',
        'decompressed_dtype': 'float32',
        'compress_expr': 'round(z)',
        'decompress_expr': 'float(z_c)',
        'precision': {
            'rmse': 0.5,
            'max_error': 1.0,
            'mean_error': 0.0,
            'fraction_nonfinite': 0.0,
            'data_range': (0, 100),
        },
        'monitor': {
            'type': 'function',
            'func': lambda p: p['rmse'] / (p['data_range'][1] - p['data_range'][0]),
            'threshold': 0.001,
            'label': 'RMSE/range'
        }
    }
    
    return adf


class TestCompressionSelection:
    """Test select_compression functionality."""
    
    def test_select_all(self, adf_with_compression):
        """Test selecting all compression entries."""
        result = adf_with_compression.select_compression()
        assert len(result) == 5
        assert set(result) == {'no_monitor', 'abs_pass', 'abs_fail', 'rel_pass', 'func_fail'}
    
    def test_select_compressed_only(self, adf_with_compression):
        """Test filtering by compressed state."""
        result = adf_with_compression.select_compression(only_compressed=True)
        assert len(result) == 4
        assert 'rel_pass' not in result  # decompressed
    
    def test_select_decompressed_only(self, adf_with_compression):
        """Test filtering by decompressed state."""
        result = adf_with_compression.select_compression(only_decompressed=True)
        assert len(result) == 1
        assert result == ['rel_pass']
    
    def test_select_failed_only(self, adf_with_compression):
        """Test filtering by monitor failure."""
        result = adf_with_compression.select_compression(only_failed=True)
        assert len(result) == 2
        assert set(result) == {'abs_fail', 'func_fail'}
    
    def test_select_pattern(self, adf_with_compression):
        """Test pattern filtering."""
        result = adf_with_compression.select_compression(pattern=r'abs.*')
        assert len(result) == 2
        assert set(result) == {'abs_pass', 'abs_fail'}
    
    def test_select_names(self, adf_with_compression):
        """Test explicit name filtering."""
        result = adf_with_compression.select_compression(names=['abs_pass', 'rel_pass'])
        assert len(result) == 2
        assert set(result) == {'abs_pass', 'rel_pass'}
    
    def test_mutually_exclusive_filters(self, adf_with_compression):
        """Test that compressed/decompressed filters are mutually exclusive."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            adf_with_compression.select_compression(
                only_compressed=True, 
                only_decompressed=True
            )
    
    def test_invalid_pattern(self, adf_with_compression):
        """Test invalid regex pattern."""
        with pytest.raises(ValueError, match="Invalid regex"):
            adf_with_compression.select_compression(pattern=r'[invalid')


class TestMonitorChecks:
    """Test monitor failure detection."""
    
    def test_absolute_monitor_pass(self, adf_with_compression):
        """Test absolute monitor that passes threshold."""
        info = adf_with_compression.compression_info['abs_pass']
        assert not adf_with_compression._check_monitor_failed(info)
    
    def test_absolute_monitor_fail(self, adf_with_compression):
        """Test absolute monitor that fails threshold."""
        info = adf_with_compression.compression_info['abs_fail']
        assert adf_with_compression._check_monitor_failed(info)
    
    def test_relative_monitor_pass(self, adf_with_compression):
        """Test relative monitor that passes threshold."""
        info = adf_with_compression.compression_info['rel_pass']
        assert not adf_with_compression._check_monitor_failed(info)
    
    def test_function_monitor_fail(self, adf_with_compression):
        """Test function-based monitor that fails threshold."""
        info = adf_with_compression.compression_info['func_fail']
        assert adf_with_compression._check_monitor_failed(info)
    
    def test_no_monitor(self, adf_with_compression):
        """Test entry without monitor never fails."""
        info = adf_with_compression.compression_info['no_monitor']
        assert not adf_with_compression._check_monitor_failed(info)


class TestMonitorValues:
    """Test monitor value computation."""
    
    def test_absolute_monitor_value(self, adf_with_compression):
        """Test computing absolute monitor value."""
        info = adf_with_compression.compression_info['abs_fail']
        label, value = adf_with_compression._get_monitor_value(info)
        assert label == 'absolute'
        assert value == 0.5
    
    def test_relative_monitor_value(self, adf_with_compression):
        """Test computing relative monitor value."""
        info = adf_with_compression.compression_info['rel_pass']
        label, value = adf_with_compression._get_monitor_value(info)
        assert label == 'relative'
        assert abs(value - 0.0005) < 1e-6  # 0.05/100
    
    def test_function_monitor_value(self, adf_with_compression):
        """Test computing function-based monitor value."""
        info = adf_with_compression.compression_info['func_fail']
        label, value = adf_with_compression._get_monitor_value(info)
        assert label == 'RMSE/range'
        assert abs(value - 0.005) < 1e-6  # 0.5/100
    
    def test_no_monitor_value(self, adf_with_compression):
        """Test entry without monitor returns None."""
        info = adf_with_compression.compression_info['no_monitor']
        label, value = adf_with_compression._get_monitor_value(info)
        assert label is None
        assert value is None


class TestDescribeCompression:
    """Test describe_compression functionality."""
    
    def test_describe_all(self, adf_with_compression, capsys):
        """Test describing all compression entries."""
        adf_with_compression.describe_compression(verbosity=0x01)
        captured = capsys.readouterr()
        assert 'no_monitor' in captured.out
        assert 'abs_pass' in captured.out
        assert 'Total: 5 columns' in captured.out
    
    def test_describe_failed_only(self, adf_with_compression, capsys):
        """Test describing only failed entries."""
        adf_with_compression.describe_compression(only_failed=True)
        captured = capsys.readouterr()
        assert 'abs_fail' in captured.out
        assert 'func_fail' in captured.out
        assert '^ Failed:' in captured.out
        assert 'Total: 2 columns' in captured.out
    
    def test_describe_as_dict(self, adf_with_compression):
        """Test returning result as dictionary."""
        result = adf_with_compression.describe_compression(as_dict=True)
        assert len(result) == 5
        assert 'abs_fail' in result
        assert result['abs_fail']['state'] == 'compressed'
    
    def test_describe_failed_as_dict(self, adf_with_compression):
        """Test returning only failed as dictionary."""
        result = adf_with_compression.describe_compression(only_failed=True, as_dict=True)
        assert len(result) == 2
        assert 'abs_fail' in result
        assert 'func_fail' in result
        assert result['abs_fail']['monitor_failed'] is True
        assert result['abs_fail']['monitor_label'] == 'absolute'
    
    def test_describe_with_pattern(self, adf_with_compression, capsys):
        """Test describing with pattern filter."""
        adf_with_compression.describe_compression(pattern=r'abs.*', verbosity=0x01)
        captured = capsys.readouterr()
        assert 'abs_pass' in captured.out
        assert 'abs_fail' in captured.out
        assert 'no_monitor' not in captured.out
        assert 'Total: 2 columns' in captured.out
    
    def test_verbosity_flags(self, adf_with_compression, capsys):
        """Test different verbosity levels."""
        # CORE only
        adf_with_compression.describe_compression(names=['abs_fail'], verbosity=0x01)
        captured = capsys.readouterr()
        assert 'abs_fail' in captured.out
        assert 'Compress:' not in captured.out
        
        # CORE + EXPR
        adf_with_compression.describe_compression(names=['abs_fail'], verbosity=0x03)
        captured = capsys.readouterr()
        assert 'Compress:' in captured.out
        assert 'Precision:' not in captured.out
        
        # ALL
        adf_with_compression.describe_compression(names=['abs_fail'], verbosity=0x0F)
        captured = capsys.readouterr()
        assert 'Compress:' in captured.out
        assert 'Precision:' in captured.out


class TestCompressionConstants:
    """Test compression verbosity constants."""
    
    def test_verbosity_constants_defined(self):
        """Test that verbosity constants are defined."""
        assert hasattr(AliasDataFrame, 'COMPRESS_SHOW_CORE')
        assert hasattr(AliasDataFrame, 'COMPRESS_SHOW_EXPR')
        assert hasattr(AliasDataFrame, 'COMPRESS_SHOW_PREC')
        assert hasattr(AliasDataFrame, 'COMPRESS_SHOW_STATS')
        assert hasattr(AliasDataFrame, 'COMPRESS_SHOW_ALL')
    
    def test_verbosity_constants_values(self):
        """Test verbosity constant values."""
        assert AliasDataFrame.COMPRESS_SHOW_CORE == 0x01
        assert AliasDataFrame.COMPRESS_SHOW_EXPR == 0x02
        assert AliasDataFrame.COMPRESS_SHOW_PREC == 0x04
        assert AliasDataFrame.COMPRESS_SHOW_STATS == 0x08
        assert AliasDataFrame.COMPRESS_SHOW_ALL == 0x0F


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
