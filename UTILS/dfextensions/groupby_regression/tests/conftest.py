"""
Pytest configuration for groupby_regression tests.
Phase 13.7.GB — registers feature and layer markers.
"""
import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "feature(id): capability feature ID from feature_taxonomy.py"
    )
    config.addinivalue_line(
        "markers", "layer(name): test quality layer — invariance|integration|smoke|validation|performance"
    )
