"""
pytest configuration for RDataFrameDSL tests.
"""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "future: marks tests for future phase techniques (Phase 7+)"
    )
