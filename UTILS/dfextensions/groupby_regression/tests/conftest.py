"""
Pytest configuration for groupby_regression tests.
Phase 13.7.GB — registers feature and layer markers.
"""
import pytest
import pathlib
import hashlib


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


def _clear_stale_numba_cache():
    """Clear Numba cache if source .py files have changed.

    Numba's cache=True stores compiled functions in __pycache__/*.nbi/nbc files.
    These embed the module's fully-qualified name at cache time. If the module
    is later imported under a different name (e.g. after package restructuring),
    the cache entries fail with ModuleNotFoundError on unpickle.

    Strategy: compute a hash of all .py source files in the package directory.
    Store it in __pycache__/.numba_source_hash. If the hash changes (or the
    file doesn't exist), delete all .nbi/.nbc files and update the hash.
    This runs once per session, before any imports trigger Numba compilation.
    """
    pkg_dir = pathlib.Path(__file__).parent
    cache_dir = pkg_dir / "__pycache__"

    # Collect hashes of all .py files (sorted for determinism)
    py_files = sorted(pkg_dir.glob("*.py"))
    hasher = hashlib.sha256()
    for f in py_files:
        hasher.update(f.name.encode())
        hasher.update(f.stat().st_mtime_ns.to_bytes(8, 'little'))
        hasher.update(f.stat().st_size.to_bytes(8, 'little'))
    current_hash = hasher.hexdigest()[:32]

    # Check stored hash
    hash_file = cache_dir / ".numba_source_hash"
    try:
        stored_hash = hash_file.read_text().strip()
    except (FileNotFoundError, OSError):
        stored_hash = ""

    if stored_hash == current_hash:
        return  # Cache is fresh

    # Source changed — clear Numba cache files
    if cache_dir.exists():
        nbi_files = list(cache_dir.glob("*.nbi"))
        nbc_files = list(cache_dir.glob("*.nbc"))
        for f in nbi_files + nbc_files:
            try:
                f.unlink()
            except OSError:
                pass
        if nbi_files or nbc_files:
            print(f"[conftest] Cleared {len(nbi_files)+len(nbc_files)} stale "
                  f"Numba cache files (source hash changed)")

    # Update hash
    cache_dir.mkdir(exist_ok=True)
    hash_file.write_text(current_hash + "\n")


# Run cache check at import time (before test collection triggers Numba)
_clear_stale_numba_cache()
