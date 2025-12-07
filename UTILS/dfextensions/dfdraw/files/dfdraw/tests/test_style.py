"""
Tests for dfdraw style system.
"""

import pytest
import json
import tempfile
from pathlib import Path

from dfdraw import (
    get_style, set_style, save_style, load_style, 
    list_styles, DEFAULT_STYLE
)


class TestDefaultStyle:
    """Test default style configuration."""
    
    def test_default_style_has_required_keys(self):
        """Default style should have all essential keys."""
        required = [
            "figure.figsize", "font.size", "scatter.alpha",
            "hist.bins", "stats.show", "colors.palette"
        ]
        for key in required:
            assert key in DEFAULT_STYLE, f"Missing key: {key}"
    
    def test_get_style_returns_copy(self):
        """get_style should return a copy, not the original."""
        style1 = get_style()
        style1["font.size"] = 999
        style2 = get_style()
        assert style2["font.size"] != 999


class TestSetStyle:
    """Test set_style function."""
    
    def setup_method(self):
        """Reset to default before each test."""
        set_style(None)
    
    def test_set_style_none_resets(self):
        """set_style(None) should reset to defaults."""
        set_style({"font.size": 99})
        set_style(None)
        assert get_style()["font.size"] == DEFAULT_STYLE["font.size"]
    
    def test_set_style_predefined(self):
        """set_style with predefined name should work."""
        set_style("publication")
        style = get_style()
        assert style["figure.dpi"] == 150  # Publication default
    
    def test_set_style_custom_dict(self):
        """set_style with dict should merge with defaults."""
        set_style({"font.size": 20, "scatter.alpha": 0.5})
        style = get_style()
        assert style["font.size"] == 20
        assert style["scatter.alpha"] == 0.5
        # Other defaults should remain
        assert style["hist.bins"] == DEFAULT_STYLE["hist.bins"]
    
    def test_set_style_invalid_name_raises(self):
        """set_style with unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown style"):
            set_style("nonexistent_style")
    
    def test_set_style_invalid_key_raises(self):
        """set_style with invalid key should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown style keys"):
            set_style({"invalid.key": 123})
    
    def test_set_style_invalid_type_raises(self):
        """set_style with invalid type should raise TypeError."""
        with pytest.raises(TypeError):
            set_style(123)


class TestListStyles:
    """Test list_styles function."""
    
    def test_list_styles_returns_list(self):
        """list_styles should return list of strings."""
        styles = list_styles()
        assert isinstance(styles, list)
        assert "default" in styles
        assert "publication" in styles
        assert "presentation" in styles


class TestSaveLoadStyle:
    """Test style persistence."""
    
    def setup_method(self):
        set_style(None)
    
    def test_save_and_load_style(self):
        """Save and load should preserve style."""
        set_style({"font.size": 42, "scatter.alpha": 0.3})
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_style.json"
            save_style(path)
            
            # Reset and reload
            set_style(None)
            assert get_style()["font.size"] != 42
            
            load_style(path)
            assert get_style()["font.size"] == 42
            assert get_style()["scatter.alpha"] == 0.3
    
    def test_saved_style_is_valid_json(self):
        """Saved style should be valid JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_style.json"
            save_style(path)
            
            with open(path) as f:
                data = json.load(f)
            
            assert isinstance(data, dict)
            assert "font.size" in data
