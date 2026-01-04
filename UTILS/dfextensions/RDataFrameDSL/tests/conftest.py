"""
pytest configuration for RDataFrameDSL tests.

Phase 12: Adds synthetic data fixtures for draw integration testing.
Phase 13.2.1.DSL: Adds invariant testing fixtures.
"""

import pytest
import numpy as np


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "future: marks tests for future phase techniques (Phase 7+)"
    )
    config.addinivalue_line(
        "markers", "requires_root: mark test as requiring ROOT"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "invariant: mark test as invariant-based correctness test"
    )


# =============================================================================
# Phase 12.1: Synthetic Data Fixtures
# =============================================================================

@pytest.fixture
def synthetic_scalar_rdf():
    """Simple scalar RDataFrame for basic tests."""
    ROOT = pytest.importorskip("ROOT")
    
    rdf = ROOT.RDataFrame(1000)
    rdf = rdf.Define("pt", "gRandom->Gaus(10, 2)")
    rdf = rdf.Define("eta", "gRandom->Uniform(-2, 2)")
    rdf = rdf.Define("phi", "gRandom->Uniform(-3.14159, 3.14159)")
    rdf = rdf.Define("isOK", "abs(eta) < 1.0")
    rdf = rdf.Define("charge", "gRandom->Rndm() > 0.5 ? 1 : -1")
    
    return rdf


@pytest.fixture
def synthetic_track_cluster_rdf():
    """
    RDataFrame emulating Track → Cluster relationship for Phase 12.2.
    
    Structure:
    - Per event: variable number of tracks (RVec)
    - Per event: variable number of clusters (RVec)
    - Index mapping: trackClusterFirst, trackClusterN
    
    This matches the TPC calibration pattern:
        TrackDataCompact.idxFirstResidual → UnbinnedResid
    """
    ROOT = pytest.importorskip("ROOT")
    
    # Need to compile helper functions for complex RVec generation
    ROOT.gInterpreter.Declare("""
    #ifndef SYNTH_DATA_HELPERS_DEFINED
    #define SYNTH_DATA_HELPERS_DEFINED
    
    #include <random>
    
    // Thread-local random generator for reproducibility
    inline std::mt19937& getSynthGen() {
        thread_local std::mt19937 gen(42);
        return gen;
    }
    
    inline ROOT::RVec<float> generateGausVec(int n, float mean, float sigma) {
        auto& gen = getSynthGen();
        std::normal_distribution<float> dist(mean, sigma);
        ROOT::RVec<float> v(n);
        for (auto& x : v) x = dist(gen);
        return v;
    }
    
    inline ROOT::RVec<float> generateUniformVec(int n, float low, float high) {
        auto& gen = getSynthGen();
        std::uniform_real_distribution<float> dist(low, high);
        ROOT::RVec<float> v(n);
        for (auto& x : v) x = dist(gen);
        return v;
    }
    
    inline ROOT::RVec<bool> generateBoolVec(int n, float prob_true) {
        auto& gen = getSynthGen();
        std::uniform_real_distribution<float> dist(0, 1);
        ROOT::RVec<bool> v(n);
        for (auto& x : v) x = (dist(gen) < prob_true);
        return v;
    }
    
    inline ROOT::RVec<int> generateClusterFirstIdx(int nTracks) {
        auto& gen = getSynthGen();
        std::poisson_distribution<int> dist(7);  // ~7 clusters per track
        ROOT::RVec<int> firstIdx(nTracks);
        int idx = 0;
        for (int i = 0; i < nTracks; i++) {
            firstIdx[i] = idx;
            idx += 3 + dist(gen);  // 3-15 clusters per track
        }
        return firstIdx;
    }
    
    inline int computeTotalClusters(const ROOT::RVec<int>& firstIdx, int nTracks) {
        if (nTracks == 0) return 0;
        // Estimate: last track starts at firstIdx[last], add average clusters
        return firstIdx[nTracks-1] + 10;  // Approximate
    }
    
    inline ROOT::RVec<int> computeClusterCounts(const ROOT::RVec<int>& firstIdx, int totalClusters) {
        int n = firstIdx.size();
        ROOT::RVec<int> counts(n);
        for (int i = 0; i < n; i++) {
            counts[i] = (i < n-1) ? (firstIdx[i+1] - firstIdx[i]) : (totalClusters - firstIdx[i]);
        }
        return counts;
    }
    
    #endif
    """)
    
    rdf = ROOT.RDataFrame(500)  # 500 events
    
    # === Track-level arrays ===
    rdf = rdf.Define("nTracks", "2 + gRandom->Poisson(4)")  # 2-10 tracks per event
    rdf = rdf.Define("trackPt", "generateGausVec(nTracks, 10.0, 2.0)")
    rdf = rdf.Define("trackEta", "generateUniformVec(nTracks, -1.0, 1.0)")
    rdf = rdf.Define("trackPhi", "generateUniformVec(nTracks, -3.14159, 3.14159)")
    rdf = rdf.Define("trackIsOK", "generateBoolVec(nTracks, 0.7)")  # 70% good tracks
    rdf = rdf.Define("trackMp4", "generateGausVec(nTracks, 0.0, 1.0)")  # mP4 parameter
    
    # === Index mapping: track → cluster ===
    rdf = rdf.Define("trackClusterFirst", "generateClusterFirstIdx(nTracks)")
    rdf = rdf.Define("nClusters", "computeTotalClusters(trackClusterFirst, nTracks)")
    rdf = rdf.Define("trackClusterN", "computeClusterCounts(trackClusterFirst, nClusters)")
    
    # === Cluster-level arrays ===
    rdf = rdf.Define("clusterDy", "generateGausVec(nClusters, 0.0, 0.5)")
    rdf = rdf.Define("clusterDz", "generateGausVec(nClusters, 0.0, 0.3)")
    rdf = rdf.Define("clusterY", "generateUniformVec(nClusters, -50.0, 50.0)")
    rdf = rdf.Define("clusterZ", "generateUniformVec(nClusters, -250.0, 250.0)")
    rdf = rdf.Define("clusterRow", "ROOT::RVec<int> v(nClusters); for(int i=0;i<nClusters;i++) v[i]=i%152; return v;")
    
    # === Derived columns (like calibration schema) ===
    rdf = rdf.Define("trackIsGood", "abs(trackMp4) < 1.5")  # Quality cut
    rdf = rdf.Define("clusterIsEdge", "abs(clusterY) > 45.0")  # Edge flag
    
    return rdf


@pytest.fixture
def track_cluster_schema():
    """Schema matching synthetic_track_cluster_rdf."""
    return {
        # Scalars
        "nTracks": "int",
        "nClusters": "int",
        # Track arrays
        "trackPt": "RVec<float>",
        "trackEta": "RVec<float>",
        "trackPhi": "RVec<float>",
        "trackIsOK": "RVec<bool>",
        "trackMp4": "RVec<float>",
        "trackIsGood": "RVec<bool>",
        "trackClusterFirst": "RVec<int>",
        "trackClusterN": "RVec<int>",
        # Cluster arrays
        "clusterDy": "RVec<float>",
        "clusterDz": "RVec<float>",
        "clusterY": "RVec<float>",
        "clusterZ": "RVec<float>",
        "clusterRow": "RVec<int>",
        "clusterIsEdge": "RVec<bool>",
    }


@pytest.fixture
def scalar_schema():
    """Schema matching synthetic_scalar_rdf."""
    return {
        "pt": "double",
        "eta": "double",
        "phi": "double",
        "isOK": "bool",
        "charge": "int",
    }


# =============================================================================
# Phase 13.2.1.DSL: Invariant Testing Fixtures
# =============================================================================

from .invariant_schema import (
    TOLERANCE,
    DEFAULT_N_EVENTS,
    DEFAULT_SEED,
    INVARIANT_SCHEMA,
)


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
        from dsl_compiler import DSLCompiler
    except ImportError:
        try:
            from ..dsl_compiler import DSLCompiler
        except ImportError:
            pytest.skip("DSLCompiler not available")
    
    # Use the DSL schema (excludes C-array branches which are TTree-specific)
    dsl_schema = {k: v for k, v in INVARIANT_SCHEMA.items() 
                  if not k.startswith("arr_") and k != "n_arr" and k != "idx" and k != "picked"}
    
    return DSLCompiler(dsl_schema)
