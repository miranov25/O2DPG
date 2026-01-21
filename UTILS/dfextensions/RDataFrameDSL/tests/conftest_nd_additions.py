"""
Conftest additions for Phase 13.6.C N-D slicing tests.

=============================================================================
WHAT: Pytest fixtures for N-D slicing invariance tests
WHY:  Provide deterministic test data with exact-value validation
WHO:  Used by test_invariance_nd.py, test_invariance_combinations.py
=============================================================================

Usage:
    # Copy content to tests/conftest.py OR import:
    from conftest_nd_additions import *

Phase: 13.6.C
Approved: PROPOSAL_ND_GENERATORS_v1.2
Phase 13.6.D+: Added ROOT introspection for automatic method discovery
Phase 13.6.D++: Fixed pragma registration order for custom classes
"""

import pytest
import logging

# Set up logging for debugging
logger = logging.getLogger(__name__)

# Phase 13.6.D+: Import ROOT introspection for automatic method discovery
try:
    from RDataFrameDSL.root_introspection import discover_class_methods
    _INTROSPECTION_AVAILABLE = True
except ImportError:
    _INTROSPECTION_AVAILABLE = False
    discover_class_methods = None


# =============================================================================
# Phase 13.6.D++: Centralized Pragma Registration
# =============================================================================

_CUSTOM_CLASS_PRAGMAS_REGISTERED = False

def ensure_custom_class_pragmas():
    """
    Ensure custom class pragmas are registered exactly once.
    
    Phase 13.6.D++: CRITICAL - Must be called BEFORE:
    1. Creating ROOT files with custom classes
    2. Opening ROOT files with custom classes
    3. Creating RDataFrame with custom classes
    
    Order matters! ROOT needs dictionaries before it can serialize/deserialize
    RVec<ToyCluster> members inside ToyTrack.
    
    Phase 13.6.D+++: CRITICAL FIX - Pragmas must be registered BEFORE
    register_custom_classes() is called, because gInterpreter.Declare()
    triggers TStreamerInfo::Build which needs the RVec<ToyCluster> dictionary.
    """
    global _CUSTOM_CLASS_PRAGMAS_REGISTERED
    
    if _CUSTOM_CLASS_PRAGMAS_REGISTERED:
        logger.debug("Custom class pragmas already registered, skipping")
        return
    
    import ROOT
    
    # Phase 13.6.D+++: Register pragmas FIRST, before class declarations
    # CRITICAL ORDER: RVec<ToyCluster> MUST be registered BEFORE ToyTrack
    # because ToyTrack contains a member 'fClusters' of type RVec<ToyCluster>
    # AND the pragmas must be processed BEFORE gInterpreter.Declare() in
    # register_custom_classes(), or TStreamerInfo::Build will fail.
    pragmas = [
        '#pragma link C++ class ToyCluster+;',
        '#pragma link C++ class ROOT::VecOps::RVec<ToyCluster>+;',  # Before ToyTrack!
        '#pragma link C++ class ToyTrack+;',
        '#pragma link C++ class ROOT::VecOps::RVec<ToyTrack>+;',
    ]
    
    logger.info("Registering ROOT pragmas for custom classes...")
    for pragma in pragmas:
        logger.debug(f"  Processing: {pragma}")
        ROOT.gInterpreter.ProcessLine(pragma)
    
    # NOW register custom classes (C++ declarations) - AFTER pragmas
    if _ND_GENERATORS_AVAILABLE:
        logger.info("Registering custom classes (ToyTrack, ToyCluster)...")
        register_custom_classes()
    
    _CUSTOM_CLASS_PRAGMAS_REGISTERED = True
    logger.info("Custom class pragmas registered successfully")


# =============================================================================
# Helper Functions (must be defined before markers)
# =============================================================================

def _root_available() -> bool:
    """Check if ROOT is available."""
    try:
        import ROOT
        return True
    except ImportError:
        return False


def _nd_generators_available() -> bool:
    """Check if N-D generators are available."""
    try:
        from toy_nd import generate_nd_2d_dict
        return True
    except ImportError:
        try:
            from generators.toy_nd import generate_nd_2d_dict
            return True
        except ImportError:
            return False


# =============================================================================
# Skip Markers
# =============================================================================

requires_root = pytest.mark.skipif(
    not _root_available(),
    reason="ROOT not available"
)

requires_nd_generators = pytest.mark.skipif(
    not _nd_generators_available(),
    reason="N-D generators not available"
)


# =============================================================================
# Safe Imports
# =============================================================================

_ND_GENERATORS_AVAILABLE = False

try:
    # Try local import first (when running from work directory)
    from toy_nd import (
        generate_nd_2d_dict,
        generate_nd_3d_dict,
        generate_nd_2d_root,
        generate_nd_3d_root,
        generate_custom_class_root,
        register_custom_classes,
        ND_2D_SCHEMA,
        ND_3D_SCHEMA,
        CUSTOM_CLASS_SCHEMA,
        cluster_value,
        hit_value,
        track_pt_value,
        get_expected_2d_slice,
        get_expected_2d_flat,
        get_expected_3d_slice,
        get_expected_3d_flat,
        get_expected_sum,
        get_expected_track_pt,
        get_expected_cluster_r,
        SIZE_CONFIGS,
        PYTHAGOREAN_TRIPLES,
    )
    _ND_GENERATORS_AVAILABLE = True
except ImportError:
    try:
        # Try generators subdirectory import
        from generators.toy_nd import (
            generate_nd_2d_dict,
            generate_nd_3d_dict,
            generate_nd_2d_root,
            generate_nd_3d_root,
            generate_custom_class_root,
            register_custom_classes,
            ND_2D_SCHEMA,
            ND_3D_SCHEMA,
            CUSTOM_CLASS_SCHEMA,
            cluster_value,
            hit_value,
            track_pt_value,
            get_expected_2d_slice,
            get_expected_2d_flat,
            get_expected_3d_slice,
            get_expected_3d_flat,
            get_expected_sum,
            get_expected_track_pt,
            get_expected_cluster_r,
            SIZE_CONFIGS,
            PYTHAGOREAN_TRIPLES,
        )
        _ND_GENERATORS_AVAILABLE = True
    except ImportError:
        # Provide None placeholders
        generate_nd_2d_dict = None
        generate_nd_3d_dict = None
        generate_nd_2d_root = None
        generate_nd_3d_root = None
        generate_custom_class_root = None
        register_custom_classes = None
        ND_2D_SCHEMA = None
        ND_3D_SCHEMA = None
        CUSTOM_CLASS_SCHEMA = None
        cluster_value = None
        hit_value = None
        track_pt_value = None
        get_expected_2d_slice = None
        get_expected_2d_flat = None
        get_expected_3d_slice = None
        get_expected_3d_flat = None
        get_expected_sum = None
        get_expected_track_pt = None
        get_expected_cluster_r = None
        SIZE_CONFIGS = None
        PYTHAGOREAN_TRIPLES = None


# =============================================================================
# Schema Fixtures
# =============================================================================

@pytest.fixture
def nd_2d_schema():
    """Schema for 2D N-D test data."""
    if _ND_GENERATORS_AVAILABLE:
        return ND_2D_SCHEMA.copy()
    return {
        'event_id': 'long',
        'n_tracks': 'int',
        'event_weight': 'double',  # Phase 13.6.C: scalar for broadcast tests
        'track_pt': 'RVec<double>',
        'track_eta': 'RVec<double>',
        'cluster_Q': 'RVec<RVec<double>>',
        'cluster_x': 'RVec<RVec<double>>',
        'cluster_y': 'RVec<RVec<double>>',
    }


@pytest.fixture
def nd_3d_schema():
    """Schema for 3D N-D test data."""
    if _ND_GENERATORS_AVAILABLE:
        return ND_3D_SCHEMA.copy()
    return {
        'event_id': 'long',
        'n_tracks': 'int',
        'event_weight': 'double',  # Phase 13.6.C: scalar for broadcast tests
        'track_pt': 'RVec<double>',
        'track_eta': 'RVec<double>',
        'cluster_Q': 'RVec<RVec<double>>',
        'hit_E': 'RVec<RVec<RVec<double>>>',
        'hit_t': 'RVec<RVec<RVec<double>>>',
    }


@pytest.fixture
def custom_class_schema():
    """
    Schema for custom class test data.
    
    Phase 13.6.D: Added _pragmas for ROOT dictionary registration.
    Phase 13.6.D+: Auto-discover method signatures using ROOT introspection.
    Phase 13.6.D++: Ensure pragmas are registered before introspection.
    """
    # Phase 13.6.D++: Ensure pragmas registered before introspection
    if _ND_GENERATORS_AVAILABLE and _root_available():
        ensure_custom_class_pragmas()
    
    # Auto-discover method signatures if introspection available
    if _INTROSPECTION_AVAILABLE and _ND_GENERATORS_AVAILABLE:
        try:
            logger.debug("Auto-discovering ToyTrack methods...")
            toy_track_methods = discover_class_methods('ToyTrack', verbose=False)
            logger.debug(f"  Found {len(toy_track_methods)} ToyTrack methods")
            
            logger.debug("Auto-discovering ToyCluster methods...")
            toy_cluster_methods = discover_class_methods('ToyCluster', verbose=False)
            logger.debug(f"  Found {len(toy_cluster_methods)} ToyCluster methods")
            
            return {
                'event_id': 'long',
                'tracks': 'RVec<ToyTrack>',
                '_pragmas': [
                    # CRITICAL ORDER: RVec<ToyCluster> MUST be before ToyTrack
                    '#pragma link C++ class ToyCluster+;',
                    '#pragma link C++ class ROOT::VecOps::RVec<ToyCluster>+;',
                    '#pragma link C++ class ToyTrack+;',
                    '#pragma link C++ class ROOT::VecOps::RVec<ToyTrack>+;',
                ],
                '_methods': {
                    'ToyTrack': toy_track_methods,      # Auto-discovered!
                    'ToyCluster': toy_cluster_methods,  # Auto-discovered!
                }
            }
        except Exception as e:
            # Fall back to schema without _methods if introspection fails
            logger.warning(f"Introspection failed: {e}")
            pass
    
    # Fallback: Return schema without _methods (for compatibility)
    if _ND_GENERATORS_AVAILABLE:
        schema = CUSTOM_CLASS_SCHEMA.copy()
        # Add _pragmas if not present
        if '_pragmas' not in schema:
            schema['_pragmas'] = [
                # CRITICAL ORDER: RVec<ToyCluster> MUST be before ToyTrack
                '#pragma link C++ class ToyCluster+;',
                '#pragma link C++ class ROOT::VecOps::RVec<ToyCluster>+;',
                '#pragma link C++ class ToyTrack+;',
                '#pragma link C++ class ROOT::VecOps::RVec<ToyTrack>+;',
            ]
        return schema
    
    return {
        'event_id': 'long',
        'tracks': 'RVec<ToyTrack>',
        '_pragmas': [
            # CRITICAL ORDER: RVec<ToyCluster> MUST be before ToyTrack
            '#pragma link C++ class ToyCluster+;',
            '#pragma link C++ class ROOT::VecOps::RVec<ToyCluster>+;',
            '#pragma link C++ class ToyTrack+;',
            '#pragma link C++ class ROOT::VecOps::RVec<ToyTrack>+;',
        ]
    }


# =============================================================================
# Tier 1: Dict Fixtures (No ROOT dependency)
# =============================================================================

@pytest.fixture
def nd_2d_dict():
    """
    2D cluster data as dict (100 events).
    
    What: Pure Python dict with RVec-like nested lists
    Why:  Tier 1 unit tests without ROOT dependency
    Who:  Parser tests, transformer tests
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return generate_nd_2d_dict(size='S')


@pytest.fixture
def nd_2d_dict_small():
    """
    2D cluster data as dict (2 events only).
    
    What: Minimal 2D dict for quick tests
    Why:  Fast iteration, exact value checking
    Who:  Quick unit tests
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return generate_nd_2d_dict(size='S', n_events=2)


@pytest.fixture
def nd_3d_dict():
    """
    3D hit data as dict (100 events).
    
    What: Pure Python dict with 3-level nesting
    Why:  Tier 1 unit tests for 3D slicing
    Who:  3D flatten tests, parser tests
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return generate_nd_3d_dict(size='S')


@pytest.fixture
def nd_3d_dict_small():
    """
    3D hit data as dict (2 events only).
    
    What: Small 3D dict for quick tests
    Why:  Fast iteration
    Who:  Quick 3D validation
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return generate_nd_3d_dict(size='S', n_events=2)


# =============================================================================
# Tier 3: ROOT File Fixtures - Size S (100 events)
# =============================================================================

@pytest.fixture(scope="session")
def nd_2d_root_file_S(tmp_path_factory):
    """
    ROOT file with 2D structure, size S (100 events).
    
    What: TTree with RVec<RVec<double>> branches
    Why:  Tier 3 end-to-end tests
    Who:  Integration tests, invariance validation
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    if not _root_available():
        pytest.skip("ROOT not available")
    
    tmpdir = tmp_path_factory.mktemp("nd_data")
    filename = str(tmpdir / "toy_nd_2d_S.root")
    logger.info(f"Creating 2D ROOT file: {filename}")
    return generate_nd_2d_root(filename, size='S')


@pytest.fixture(scope="session")
def nd_3d_root_file_S(tmp_path_factory):
    """
    ROOT file with 3D structure, size S (100 events).
    
    What: TTree with RVec<RVec<RVec<double>>> branches
    Why:  Tier 3 end-to-end 3D tests
    Who:  3D integration tests
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    if not _root_available():
        pytest.skip("ROOT not available")
    
    tmpdir = tmp_path_factory.mktemp("nd_data")
    filename = str(tmpdir / "toy_nd_3d_S.root")
    logger.info(f"Creating 3D ROOT file: {filename}")
    return generate_nd_3d_root(filename, size='S')


@pytest.fixture(scope="session")
def custom_class_root_file_S(tmp_path_factory):
    """
    ROOT file with custom class branches, size S.
    
    What: TTree with RVec<ToyTrack> containing RVec<ToyCluster>
    Why:  Test method calls (.Pt(), .getQ(), .r())
    Who:  Method call tests, computed property tests
    
    Phase 13.6.D++: CRITICAL - Pragmas must be registered BEFORE file creation!
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    if not _root_available():
        pytest.skip("ROOT not available")
    
    # Phase 13.6.D++: MUST register pragmas BEFORE creating the file
    # This ensures ROOT can properly serialize RVec<ToyCluster> members
    ensure_custom_class_pragmas()
    
    tmpdir = tmp_path_factory.mktemp("custom_class")
    filename = str(tmpdir / "toy_custom_S.root")
    logger.info(f"Creating custom class ROOT file: {filename}")
    result = generate_custom_class_root(filename, size='S')
    logger.info(f"Custom class ROOT file created: {result}")
    return result


# =============================================================================
# Tier 3: ROOT File Fixtures - Persistent (M/L for benchmarks)
# =============================================================================

@pytest.fixture(scope="session")
def nd_2d_root_file_M():
    """
    ROOT file with 2D structure, size M (1000 events), persistent.
    
    What: Larger dataset for statistical validation
    Why:  Manual inspection, TBrowser viewing
    Who:  Debugging, statistical tests
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    if not _root_available():
        pytest.skip("ROOT not available")
    
    return generate_nd_2d_root(size='M', persistent=True)


@pytest.fixture(scope="session")
def nd_2d_root_file_L():
    """
    ROOT file with 2D structure, size L (10000 events), persistent.
    
    What: Large dataset for benchmarks
    Why:  Performance testing, memory profiling
    Who:  Benchmark tests
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    if not _root_available():
        pytest.skip("ROOT not available")
    
    return generate_nd_2d_root(size='L', persistent=True)


# =============================================================================
# RDataFrame Fixtures
# =============================================================================

@pytest.fixture
def nd_2d_rdf(nd_2d_root_file_S):
    """RDataFrame with 2D cluster structure (size S)."""
    import ROOT
    logger.debug(f"Opening 2D RDataFrame from: {nd_2d_root_file_S}")
    return ROOT.RDataFrame("Events", nd_2d_root_file_S)


@pytest.fixture
def nd_3d_rdf(nd_3d_root_file_S):
    """RDataFrame with 3D hit structure (size S)."""
    import ROOT
    logger.debug(f"Opening 3D RDataFrame from: {nd_3d_root_file_S}")
    return ROOT.RDataFrame("Events", nd_3d_root_file_S)


@pytest.fixture
def custom_class_rdf(custom_class_root_file_S):
    """
    RDataFrame with custom class branches (size S).
    
    Phase 13.6.D++: Pragmas already registered by custom_class_root_file_S fixture.
    We ensure they're registered again (idempotent) before opening the file.
    """
    import ROOT
    
    # Phase 13.6.D++: Ensure pragmas are registered (idempotent)
    # The file fixture already did this, but we ensure it here too for safety
    ensure_custom_class_pragmas()
    
    logger.info(f"Opening custom class RDataFrame from: {custom_class_root_file_S}")
    rdf = ROOT.RDataFrame("Events", custom_class_root_file_S)
    
    # Debug: Check if we can access the data
    count = rdf.Count().GetValue()
    logger.info(f"Custom class RDataFrame has {count} entries")
    
    return rdf


# =============================================================================
# Validation Helper Fixtures
# =============================================================================

@pytest.fixture
def expected_2d_values():
    """
    Helper function for 2D exact-value validation.
    
    Usage:
        expected = expected_2d_values(event=0, track_slice=slice(0,2), 
                                      cluster_slice=slice(0,3), layout=data['_layout'])
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return get_expected_2d_slice


@pytest.fixture
def expected_2d_flat():
    """Helper function for flattened 2D validation."""
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return get_expected_2d_flat


@pytest.fixture
def expected_3d_values():
    """Helper function for 3D exact-value validation."""
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return get_expected_3d_slice


@pytest.fixture
def expected_3d_flat():
    """Helper function for flattened 3D validation."""
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return get_expected_3d_flat


@pytest.fixture
def expected_sum():
    """Helper function for sum invariance validation."""
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return get_expected_sum


@pytest.fixture
def cluster_value_fn():
    """
    Cluster invariant function: Q = 1000*event + 100*track + cluster
    
    Usage:
        expected_Q = cluster_value_fn(event=0, track=1, cluster=2)  # 102.0
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return cluster_value


@pytest.fixture
def hit_value_fn():
    """
    Hit invariant function: E = 10000*event + 1000*track + 100*cluster + hit
    
    Usage:
        expected_E = hit_value_fn(event=0, track=1, cluster=2, hit=3)  # 1203.0
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return hit_value


@pytest.fixture
def pythagorean_triples():
    """
    Pythagorean triples for track pt validation.
    
    Usage:
        px, py, pt = pythagorean_triples[track_idx % len(pythagorean_triples)]
        assert track.Pt() == pt  # Exact
    """
    if not _ND_GENERATORS_AVAILABLE:
        pytest.skip("N-D generators not available")
    return PYTHAGOREAN_TRIPLES
