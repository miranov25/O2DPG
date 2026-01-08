"""
Phase 13.2: pytest configuration and shared fixtures for RDataFrameDSL tests.

Provides:
- Tolerance helpers for invariant testing
- Session-scoped test data generation
- DSL compiler fixtures
- ROOT test isolation markers (Phase 13.4.D9)
"""

import pytest
import numpy as np

# Import invariant schema configuration
try:
    from .invariant_schema import (
        TOLERANCE,
        DEFAULT_N_EVENTS,
        DEFAULT_SEED,
        INVARIANT_SCHEMA,
    )
except ImportError:
    from invariant_schema import (
        TOLERANCE,
        DEFAULT_N_EVENTS,
        DEFAULT_SEED,
        INVARIANT_SCHEMA,
    )


# =============================================================================
# Tolerance Helpers
# =============================================================================

def get_tolerance(dtype: str) -> dict:
    """
    Get tolerance settings for a given dtype.
    
    Parameters:
        dtype: One of 'double', 'float', 'int', 'uint', 'bool', 'aggregation'
    
    Returns:
        Dict with 'atol' and 'rtol' keys
    """
    dtype_key = dtype.lower()
    if dtype_key in ("int32", "int64", "signed int"):
        dtype_key = "int"
    elif dtype_key in ("uint32", "uint64", "unsigned int"):
        dtype_key = "uint"
    elif dtype_key == "float32":
        dtype_key = "float"
    elif dtype_key in ("float64", "double"):
        dtype_key = "double"
    
    return TOLERANCE.get(dtype_key, TOLERANCE["double"])


def assert_invariant(actual, expected, dtype="double", context=""):
    """
    Assert that actual matches expected within dtype-appropriate tolerance.
    
    Parameters:
        actual: Computed value
        expected: Expected value (from invariant)
        dtype: Data type for tolerance selection
        context: Description for error messages
    
    Raises:
        AssertionError: If values differ beyond tolerance
    """
    tol = get_tolerance(dtype)
    
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        
        if tol["atol"] == 0 and tol["rtol"] == 0:
            # Exact comparison
            if not np.array_equal(actual, expected):
                diff_idx = np.where(actual != expected)[0]
                raise AssertionError(
                    f"Exact match failed ({context}): "
                    f"differs at indices {diff_idx[:5]}..."
                )
        else:
            # Tolerance-based comparison
            if not np.allclose(actual, expected, atol=tol["atol"], rtol=tol["rtol"]):
                diff = np.abs(actual - expected)
                max_diff_idx = np.argmax(diff)
                raise AssertionError(
                    f"Tolerance exceeded ({context}): "
                    f"max_diff={diff[max_diff_idx]:.2e} at index {max_diff_idx}"
                )
    else:
        # Scalar comparison
        if tol["atol"] == 0 and tol["rtol"] == 0:
            if actual != expected:
                raise AssertionError(
                    f"Exact match failed ({context}): actual={actual} expected={expected}"
                )
        else:
            threshold = tol["atol"] + tol["rtol"] * abs(expected)
            if abs(actual - expected) > threshold:
                raise AssertionError(
                    f"Tolerance exceeded ({context}): "
                    f"actual={actual} expected={expected} diff={abs(actual - expected):.2e}"
                )


# =============================================================================
# Session-Scoped Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory):
    """Create a session-scoped temporary directory for test data."""
    return tmp_path_factory.mktemp("invariant_test_data")


@pytest.fixture(scope="session")
def invariant_tree_path(test_data_dir):
    """
    Generate the invariant test TTree (session-scoped).
    
    Returns:
        Path to ROOT file, or None if ROOT not available
    """
    try:
        import ROOT
        from .test_data_generator import generate_invariant_tree
        
        filepath = test_data_dir / "invariant_data.root"
        generate_invariant_tree(
            str(filepath),
            n_events=DEFAULT_N_EVENTS,
            seed=DEFAULT_SEED,
        )
        return filepath
    except ImportError:
        return None


@pytest.fixture
def invariant_rdf(invariant_tree_path):
    """
    Create an RDataFrame from the invariant test tree.
    
    Yields:
        ROOT.RDataFrame or pytest.skip if ROOT not available
    """
    if invariant_tree_path is None:
        pytest.skip("ROOT not available")
    
    import ROOT
    rdf = ROOT.RDataFrame("invariants", str(invariant_tree_path))
    return rdf


@pytest.fixture(scope="session")
def small_invariant_tree_path(test_data_dir):
    """
    Generate a small invariant test TTree for quick tests.
    
    Returns:
        Path to ROOT file with 100 events
    """
    try:
        import ROOT
        from .test_data_generator import generate_invariant_tree
        
        filepath = test_data_dir / "invariant_data_small.root"
        generate_invariant_tree(
            str(filepath),
            n_events=100,
            seed=DEFAULT_SEED,
        )
        return filepath
    except ImportError:
        return None


@pytest.fixture
def dsl_compiler():
    """
    Create a DSLCompiler with the invariant schema.
    
    Returns:
        DSLCompiler instance or pytest.skip if not available
    """
    try:
        from RDataFrameDSL.dsl_compiler import DSLCompiler
    except ImportError:
        try:
            from dsl_compiler import DSLCompiler
        except ImportError:
            pytest.skip("DSLCompiler not available")
    
    # Use the DSL schema (excludes C-array branches which are TTree-specific)
    dsl_schema = {k: v for k, v in INVARIANT_SCHEMA.items() 
                  if not k.startswith("arr_") and k != "n_arr" and k != "idx" and k != "picked"}
    
    return DSLCompiler(dsl_schema)


# =============================================================================
# Fixtures for test_draw_integration.py
# =============================================================================

@pytest.fixture
def scalar_schema():
    """Schema with scalar columns for draw integration tests."""
    return {"pt": "double", "eta": "double", "phi": "double"}


@pytest.fixture
def track_cluster_schema():
    """Schema with RVec columns for draw integration tests."""
    return {
        "trackPt": "RVec<float>",
        "trackEta": "RVec<float>",
        "clusterE": "RVec<float>",
        "nTracks": "int",      # P0-2/P0-3: Include counter columns
        "nClusters": "int",
    }


@pytest.fixture
def synthetic_scalar_rdf(tmp_path, scalar_schema):
    """
    Create an RDataFrame with synthetic scalar data.
    
    Generates a ROOT file with scalar columns (pt, eta, phi).
    """
    try:
        import ROOT
    except ImportError:
        pytest.skip("ROOT not available")
    
    # Create a temporary ROOT file with scalar data
    filepath = tmp_path / "scalar_data.root"
    
    # Use RDataFrame to create synthetic data
    n_events = 100
    rdf = ROOT.RDataFrame(n_events)
    
    # Add scalar columns with generated data
    rdf_with_data = (
        rdf.Define("pt", "gRandom->Uniform(0.5, 100.0)")
           .Define("eta", "gRandom->Uniform(-2.5, 2.5)")
           .Define("phi", "gRandom->Uniform(-3.14159, 3.14159)")
    )
    
    # Save to file and re-read (ensures proper column types)
    rdf_with_data.Snapshot("tree", str(filepath))
    
    return ROOT.RDataFrame("tree", str(filepath))


@pytest.fixture
def synthetic_track_cluster_rdf(tmp_path, track_cluster_schema):
    """
    Create an RDataFrame with synthetic track/cluster data (RVec columns).
    
    Generates a ROOT file with RVec<float> columns.
    """
    try:
        import ROOT
    except ImportError:
        pytest.skip("ROOT not available")
    
    # Create a temporary ROOT file with RVec data
    filepath = tmp_path / "track_cluster_data.root"
    
    n_events = 100
    rdf = ROOT.RDataFrame(n_events)
    
    # Add RVec columns - generate variable-length vectors
    rdf_with_data = (
        rdf.Define("nTracks", "gRandom->Integer(10) + 1")  # 1-10 tracks
           .Define("trackPt", "ROOT::RVecF v(nTracks); for(auto& x : v) x = gRandom->Uniform(0.5, 50.0); return v;")
           .Define("trackEta", "ROOT::RVecF v(nTracks); for(auto& x : v) x = gRandom->Uniform(-1.0, 1.0); return v;")
           .Define("nClusters", "gRandom->Integer(20) + 1")  # 1-20 clusters
           .Define("clusterE", "ROOT::RVecF v(nClusters); for(auto& x : v) x = gRandom->Uniform(0.1, 10.0); return v;")
    )
    
    # P0-2 FIX: Save ALL columns including nTracks and nClusters
    rdf_with_data.Snapshot("tree", str(filepath), {"trackPt", "trackEta", "clusterE", "nTracks", "nClusters"})
    
    return ROOT.RDataFrame("tree", str(filepath))


# =============================================================================
# Phase 13.4.D9: ROOT Test Isolation
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "root_serial: mark test as requiring ROOT (runs serially, deselect with -m 'not root_serial')"
    )



def pytest_collection_modifyitems(config, items):
    """
    Auto-mark tests that use ROOT for serial execution.
    
    Tests are marked as root_serial if:
    1. Test file name contains patterns indicating ROOT usage
    2. Test is not in the parallel-safe list
    """
    # Patterns that indicate ROOT usage
    root_patterns = [
        'test_carray_correctness',  # JIT declarations
        'test_carray_root',         # JIT declarations
        'test_root_broadcast',      # Custom Define() code
        'test_root_integration',    # Custom Define() code
    ]
    
    # Tests that are safe for parallel (no ROOT JIT)
    parallel_safe = [
        'test_d9_integration',
        'test_carray_detector',
        'test_arrow',
        'test_backend_cpp',
        'test_ir_',
        'test_parser',
        'test_type_inferrer',
        'test_schema',
        'test_generator_sanity',
    ]
    
    root_serial_marker = pytest.mark.root_serial
    
    for item in items:
        # Get test file name
        test_file = item.fspath.basename if hasattr(item, 'fspath') else str(item.path)
        
        # Check if it matches ROOT patterns
        is_root_test = any(pattern in test_file for pattern in root_patterns)
        is_parallel_safe = any(pattern in test_file for pattern in parallel_safe)
        
        # Mark ROOT tests for serial execution
        if is_root_test and not is_parallel_safe:
            item.add_marker(root_serial_marker)


@pytest.fixture(scope="session")
def root_lock():
    """
    Session-scoped lock for ROOT operations.
    
    Use this fixture in tests that need exclusive ROOT access:
    
        def test_something(root_lock):
            with root_lock:
                ROOT.gInterpreter.Declare(...)
    """
    import threading
    return threading.Lock()
