"""
dfdraw - DataFrame drawing utilities with TTree::Draw-like interface.

Usage:
    from dfdraw import DFDraw, draw, hist, profile, scatter
    
    # Class-based
    plotter = DFDraw(df)
    fig, ax, stats = plotter.draw("y:x", color="category")
    
    # Functional
    fig, ax, stats = draw(df, "y:x")
    fig, ax, stats = hist(df, "x", bins=100)
"""

from .drawer import DFDraw
from .style import (
    get_style,
    set_style,
    save_style,
    load_style,
    list_styles,
    DEFAULT_STYLE,
)

# Functional API (convenience wrappers)
def draw(data, expr, **kwargs):
    """Draw plot from DataFrame using TTree::Draw-like syntax."""
    return DFDraw(data).draw(expr, **kwargs)

def hist(data, expr, **kwargs):
    """Draw 1D histogram."""
    return DFDraw(data).hist(expr, **kwargs)

def hist2d(data, expr, **kwargs):
    """Draw 2D histogram."""
    return DFDraw(data).hist2d(expr, **kwargs)

def scatter(data, expr, **kwargs):
    """Draw scatter plot."""
    return DFDraw(data).scatter(expr, **kwargs)

def profile(data, expr, **kwargs):
    """Draw profile plot (mean of y in bins of x)."""
    return DFDraw(data).profile(expr, **kwargs)

__all__ = [
    # Class
    "DFDraw",
    # Functional
    "draw", "hist", "hist2d", "scatter", "profile",
    # Style
    "get_style", "set_style", "save_style", "load_style", "list_styles",
    "DEFAULT_STYLE",
]

__version__ = "0.1.0"
