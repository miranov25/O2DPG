"""
conftest.py — Pytest configuration and shared fixtures for RDataFrameDSL tests.

Provides:
- Tolerance helpers for invariant testing
- Session-scoped test data generation
- DSL compiler fixtures
- ROOT test isolation markers (Phase 13.4.D9)
- ALICE event generator fixtures (Phase 13.6.B)
- Toy Lorentz generator fixtures (Phase 13.6.B)
- Capability Matrix markers (Phase 13.6.B.fix)

Phases:
- 13.2: Initial pytest configuration
- 13.4.D9: ROOT test isolation
- 13.6.B: Invariance test fixtures
- 13.6.B.fix: Capability Matrix semi-automation

WARNING: This file must have exactly ONE pytest_configure function.
         Multiple definitions cause markers to be silently lost.
"""

import pytest
import numpy as np
import os
import sys

# =============================================================================
# Path Setup
# =============================================================================

_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# =============================================================================
# Import invariant schema configuration
# =============================================================================

try:
    from .invariant_schema import (
        TOLERANCE,
        DEFAULT_N_EVENTS,
        DEFAULT_SEED,
        INVARIANT_SCHEMA,
    )
except ImportError:
    try:
        from invariant_schema import (
            TOLERANCE,
            DEFAULT_N_EVENTS,
            DEFAULT_SEED,
            INVARIANT_SCHEMA,
        )
    except ImportError:
        # Provide defaults if invariant_schema not available
        TOLERANCE = {
            "double": {"atol": 1e-10, "rtol": 1e-10},
            "float": {"atol": 1e-5, "rtol": 1e-5},
            "int": {"atol": 0, "rtol": 0},
            "uint": {"atol": 0, "rtol": 0},
            "bool": {"atol": 0, "rtol": 0},
            "aggregation": {"atol": 1e-8, "rtol": 1e-8},
        }
        DEFAULT_N_EVENTS = 1000
        DEFAULT_SEED = 42
        INVARIANT_SCHEMA = {}

# =============================================================================
# Safe generator imports (Phase 13.6.B)
# =============================================================================

_GENERATORS_AVAILABLE = False
try:
    from generators.alice_events import ALICEEventGenerator, GeneratorConfig
    from generators.toy_lorentz import (
        generate_toy_lorentz_root,
        generate_toy_lorentz_dict,
        generate_toy_with_clusters
    )
    _GENERATORS_AVAILABLE = True
except ImportError:
    ALICEEventGenerator = None
    GeneratorConfig = None
    generate_toy_lorentz_root = None
    generate_toy_lorentz_dict = None
    generate_toy_with_clusters = None


# #############################################################################
#
#  PYTEST CONFIGURATION — ALL MARKERS IN ONE FUNCTION
#
#  WARNING: Python only keeps ONE pytest_configure. If you define it twice,
#           the first one is silently replaced and its markers are lost!
#
# #############################################################################

def pytest_configure(config):
    """
    Register ALL custom markers.
    
    This is the ONLY pytest_configure in this file.
    All marker registrations MUST be here.
    """
    # -------------------------------------------------------------------------
    # Phase 13.4.D9: ROOT test isolation marker
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "root_serial: mark test as requiring ROOT (runs serially, deselect with -m 'not root_serial')"
    )
    
    # -------------------------------------------------------------------------
    # Phase 13.6.B: Test type and priority markers
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers", 
        "type_a: Type A tests (Engine-level, flatten only)"
    )
    config.addinivalue_line(
        "markers", 
        "type_b: Type B tests (DSL-level, full pipeline)"
    )
    config.addinivalue_line(
        "markers", 
        "p0: Priority 0 (blocking/critical)"
    )
    config.addinivalue_line(
        "markers", 
        "p1: Priority 1 (important/required)"
    )
    config.addinivalue_line(
        "markers", 
        "p2: Priority 2 (nice to have/suggested)"
    )
    config.addinivalue_line(
        "markers", 
        "phase8: Requires Phase 8 method broadcasting"
    )
    
    # -------------------------------------------------------------------------
    # Phase 13.6.B.fix: Capability Matrix markers
    # -------------------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "feature(name): Feature ID from FEATURE_TAXONOMY (see tests/feature_taxonomy.py)"
    )
    config.addinivalue_line(
        "markers",
        "phase(id): Phase that implemented this feature (e.g., '8', '13.6', '13.6.B.fix')"
    )
    config.addinivalue_line(
        "markers",
        "limitation(id): Limitation ID from KNOWN_LIMITATIONS (e.g., 'L1')"
    )


# #############################################################################
#
#  PYTEST COLLECTION HOOK — ROOT isolation + Feature ID validation
#
# #############################################################################

def pytest_collection_modifyitems(config, items):
    """
    Hook called after test collection. Does two things:
    
    1. Auto-mark ROOT tests for serial execution (Phase 13.4.D9)
    2. Validate feature IDs with LAZY IMPORT (Phase 13.6.B.fix)
    """
    # =========================================================================
    # Part 1: ROOT Test Isolation (Phase 13.4.D9)
    # =========================================================================
    
    root_patterns = [
        'test_carray_correctness',
        'test_carray_root',
        'test_root_broadcast',
        'test_root_integration',
    ]
    
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
        test_file = item.fspath.basename if hasattr(item, 'fspath') else str(item.path)
        is_root_test = any(pattern in test_file for pattern in root_patterns)
        is_parallel_safe = any(pattern in test_file for pattern in parallel_safe)
        
        if is_root_test and not is_parallel_safe:
            item.add_marker(root_serial_marker)
    
    # =========================================================================
    # Part 2: Feature ID Validation (Phase 13.6.B.fix)
    # =========================================================================
    #
    # CRITICAL: Uses LAZY IMPORT to avoid loading taxonomy for every pytest run.
    # Only validates when a test actually uses @pytest.mark.feature.
    # This prevents conftest.py interference!
    #
    
    # Quick check: do any tests use feature markers?
    needs_validation = False
    for item in items:
        if list(item.iter_markers(name="feature")):
            needs_validation = True
            break
    
    if not needs_validation:
        # No tests use @pytest.mark.feature — skip taxonomy import entirely
        return
    
    # LAZY IMPORT: Try multiple paths
    FEATURE_TAXONOMY = None
    FEATURE_ALIASES = None
    
    import_attempts = [
        "feature_taxonomy",
        "tests.feature_taxonomy",
    ]
    
    for module_name in import_attempts:
        try:
            import importlib
            module = importlib.import_module(module_name)
            FEATURE_TAXONOMY = getattr(module, "FEATURE_TAXONOMY", None)
            FEATURE_ALIASES = getattr(module, "FEATURE_ALIASES", {})
            if FEATURE_TAXONOMY is not None:
                break
        except ImportError:
            continue
    
    if FEATURE_TAXONOMY is None:
        # Taxonomy not available — skip validation silently
        import warnings
        warnings.warn(
            "feature_taxonomy.py not found — feature marker validation skipped.",
            UserWarning
        )
        return
    
    # Build valid feature IDs
    valid_feature_ids = set(FEATURE_TAXONOMY.keys())
    if FEATURE_ALIASES:
        valid_feature_ids.update(FEATURE_ALIASES.keys())
    
    # Validate
    invalid_features = []
    for item in items:
        for marker in item.iter_markers(name="feature"):
            if marker.args:
                feature_id = marker.args[0]
                if FEATURE_ALIASES and feature_id in FEATURE_ALIASES:
                    feature_id = FEATURE_ALIASES[feature_id]
                if feature_id not in FEATURE_TAXONOMY:
                    invalid_features.append((item.nodeid, feature_id))
    
    if invalid_features:
        error_lines = ["", "=" * 70, "INVALID FEATURE IDs FOUND", "=" * 70, ""]
        for nodeid, feature_id in invalid_features:
            error_lines.append(f"  ✗ {nodeid}")
            error_lines.append(f"    Feature ID: '{feature_id}' (not in taxonomy)")
            error_lines.append("")
        error_lines.extend(["-" * 70, "Valid feature IDs:", ""])
        for fid in sorted(FEATURE_TAXONOMY.keys()):
            error_lines.append(f"  • {fid}")
        error_lines.extend(["", "See: tests/feature_taxonomy.py", "=" * 70])
        raise ValueError("\n".join(error_lines))


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
    return {"pt": "double", "eta": "double", "phi": "double", "isOK": "bool"}


@pytest.fixture
def track_cluster_schema():
    """Schema with RVec columns for draw integration tests."""
    return {
        "trackPt": "RVec<float>",
        "trackEta": "RVec<float>",
        "trackIsOK": "RVec<bool>",
        "clusterE": "RVec<float>",
        "clusterDy": "RVec<float>",
        "clusterZ": "RVec<float>",
        "nTracks": "int",
        "nClusters": "int",
    }


@pytest.fixture
def synthetic_scalar_rdf(tmp_path, scalar_schema):
    """
    Create an RDataFrame with synthetic scalar data.
    
    Generates a ROOT file with scalar columns (pt, eta, phi, isOK).
    """
    try:
        import ROOT
    except ImportError:
        pytest.skip("ROOT not available")
    
    filepath = tmp_path / "scalar_data.root"
    n_events = 100
    rdf = ROOT.RDataFrame(n_events)
    
    rdf_with_data = (
        rdf.Define("pt", "gRandom->Uniform(0.5, 100.0)")
           .Define("eta", "gRandom->Uniform(-2.5, 2.5)")
           .Define("phi", "gRandom->Uniform(-3.14159, 3.14159)")
           .Define("isOK", "abs(eta) < 1.0")
    )
    
    rdf_with_data.Snapshot("tree", str(filepath))
    return ROOT.RDataFrame("tree", str(filepath))


@pytest.fixture
def synthetic_track_cluster_rdf(tmp_path, track_cluster_schema):
    """
    Create an RDataFrame with synthetic track/cluster data (RVec columns).
    """
    try:
        import ROOT
    except ImportError:
        pytest.skip("ROOT not available")
    
    filepath = tmp_path / "track_cluster_data.root"
    n_events = 100
    rdf = ROOT.RDataFrame(n_events)
    
    rdf_with_data = (
        rdf.Define("nTracks", "gRandom->Integer(10) + 1")
           .Define("trackPt", "ROOT::RVecF v(nTracks); for(auto& x : v) x = gRandom->Uniform(0.5, 50.0); return v;")
           .Define("trackEta", "ROOT::RVecF v(nTracks); for(auto& x : v) x = gRandom->Uniform(-1.0, 1.0); return v;")
           .Define("trackIsOK", "ROOT::RVec<bool> v(nTracks); for(auto& x : v) x = gRandom->Rndm() > 0.2; return v;")
           .Define("nClusters", "gRandom->Integer(20) + 1")
           .Define("clusterE", "ROOT::RVecF v(nClusters); for(auto& x : v) x = gRandom->Uniform(0.1, 10.0); return v;")
           .Define("clusterDy", "ROOT::RVecF v(nClusters); for(auto& x : v) x = gRandom->Gaus(0, 0.1); return v;")
           .Define("clusterZ", "ROOT::RVecF v(nClusters); for(auto& x : v) x = gRandom->Uniform(-200, 200); return v;")
    )
    
    rdf_with_data.Snapshot("tree", str(filepath), 
                           {"trackPt", "trackEta", "trackIsOK", 
                            "clusterE", "clusterDy", "clusterZ",
                            "nTracks", "nClusters"})
    
    return ROOT.RDataFrame("tree", str(filepath))


# =============================================================================
# ROOT Lock Fixture (Phase 13.4.D9)
# =============================================================================

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


# =============================================================================
# ALICE Event Generator Fixtures (Phase 13.6.B)
# =============================================================================

@pytest.fixture(scope="module")
def alice_data_xs():
    """Generate XS (extra-small) ALICE test data (100 events)."""
    if not _GENERATORS_AVAILABLE:
        pytest.skip("ALICE generators not available")
    gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
    return gen.generate_dict(n_events=100)


@pytest.fixture(scope="module")
def alice_data_small():
    """Generate small ALICE test data (500 events)."""
    if not _GENERATORS_AVAILABLE:
        pytest.skip("ALICE generators not available")
    gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
    return gen.generate_dict(n_events=500)


@pytest.fixture(scope="module")
def alice_rdf(tmp_path_factory):
    """Generate ALICE ROOT file and return RDataFrame."""
    if not _GENERATORS_AVAILABLE:
        pytest.skip("ALICE generators not available")
    try:
        import ROOT
    except ImportError:
        pytest.skip("ROOT not available")
    
    tmpdir = tmp_path_factory.mktemp("alice")
    filename = str(tmpdir / "alice_test.root")
    gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
    gen.generate_tree(filename, n_events=100)
    return ROOT.RDataFrame("Events", filename)


@pytest.fixture(scope="module")
def alice_root_file(tmp_path_factory):
    """Generate ALICE ROOT file and return path."""
    if not _GENERATORS_AVAILABLE:
        pytest.skip("ALICE generators not available")
    try:
        import ROOT
    except ImportError:
        pytest.skip("ROOT not available")
    
    tmpdir = tmp_path_factory.mktemp("alice_file")
    filename = str(tmpdir / "alice_events.root")
    gen = ALICEEventGenerator(config=GeneratorConfig(seed=42))
    gen.generate_tree(filename, n_events=100)
    return filename


# =============================================================================
# Toy Lorentz Generator Fixtures (Phase 13.6.B)
# =============================================================================

@pytest.fixture(scope="module")
def toy_lorentz_file(tmp_path_factory):
    """Generate toy ROOT file with TLorentzVector."""
    if not _GENERATORS_AVAILABLE:
        pytest.skip("Toy generators not available")
    try:
        import ROOT
    except ImportError:
        pytest.skip("ROOT not available")
    
    tmpdir = tmp_path_factory.mktemp("toy")
    filename = str(tmpdir / "toy_lorentz.root")
    generate_toy_lorentz_root(filename, n_events=3, tracks_per_event=[2, 3, 2])
    return filename


@pytest.fixture(scope="module")
def toy_data():
    """Generate toy dict data."""
    if not _GENERATORS_AVAILABLE:
        pytest.skip("Toy generators not available")
    return generate_toy_lorentz_dict(n_events=3)


@pytest.fixture(scope="module")
def toy_data_with_clusters():
    """Generate toy dict data with 2D cluster structure."""
    if not _GENERATORS_AVAILABLE:
        pytest.skip("Toy generators not available")
    return generate_toy_with_clusters(n_events=3)


@pytest.fixture(scope="module")
def toy_rdf(toy_lorentz_file):
    """Return RDataFrame from toy Lorentz file."""
    try:
        import ROOT
        return ROOT.RDataFrame("Events", toy_lorentz_file)
    except ImportError:
        pytest.skip("ROOT not available")


# =============================================================================
# Schema Fixtures (Phase 13.6.B)
# =============================================================================

@pytest.fixture
def dsl_schema():
    """Default schema for ALICE data."""
    return {
        'event_id': 'long',
        'n_tracks': 'int',
        'vertex_z': 'double',
        'track_pt': 'RVec<double>',
        'track_phi': 'RVec<double>',
        'track_eta': 'RVec<double>',
        'track_px': 'RVec<double>',
        'track_py': 'RVec<double>',
        'cluster_x': 'RVec<RVec<double>>',
        'cluster_y': 'RVec<RVec<double>>',
        'cluster_Q': 'RVec<RVec<double>>',
    }


@pytest.fixture
def toy_schema():
    """Schema for toy Lorentz data."""
    return {
        'event_id': 'long',
        'n_tracks': 'int',
        'tracks': 'RVec<TLorentzVector>',
        'track_pt': 'RVec<double>',
        'track_px': 'RVec<double>',
        'track_py': 'RVec<double>',
        'track_phi': 'RVec<double>',
        'track_eta': 'RVec<double>',
    }


@pytest.fixture
def simple_range_data():
    """Simple deterministic data for range/sliding tests."""
    return {
        'event_id': np.array([0, 1, 2], dtype=np.int64),
        'n_tracks': np.array([2, 3, 2], dtype=np.int32),
        'track_pt': np.array([
            np.array([1.0, 2.0], dtype=np.float64),
            np.array([3.0, 4.0, 5.0], dtype=np.float64),
            np.array([6.0, 7.0], dtype=np.float64),
        ], dtype=object),
        'cluster_Q': np.array([
            np.array([
                np.array([10., 11., 12., 13.], dtype=np.float64),
                np.array([20., 21., 22., 23., 24.], dtype=np.float64),
            ], dtype=object),
            np.array([
                np.array([30., 31., 32.], dtype=np.float64),
                np.array([40., 41., 42., 43.], dtype=np.float64),
                np.array([50., 51.], dtype=np.float64),
            ], dtype=object),
            np.array([
                np.array([60., 61., 62., 63.], dtype=np.float64),
                np.array([70., 71., 72.], dtype=np.float64),
            ], dtype=object),
        ], dtype=object),
    }
