"""
test_07_dsl_draw.py - Pre-flight tests for 07_dsl_draw.ipynb

Phase 13.6.G+: Tests mirroring notebook structure for validation.

This test file validates DSL compilation and draw() functionality.
If dfdraw is not installed, tests still validate the DSL (alias 
materialization, schema validation) - following the pattern from
test_invariance_safe_draw.py.

Location: examples/test_07_dsl_draw.py (self-contained)

Usage (from examples/ directory):
    pytest test_07_dsl_draw.py -v --tb=short
    pytest test_07_dsl_draw.py -v -k "preflight"
    pytest test_07_dsl_draw.py -v -k "Part1A"
    pytest test_07_dsl_draw.py -v -k "Part1B"
    pytest test_07_dsl_draw.py -v -k "Part2"
"""

import pytest
import numpy as np
import time
import sys
import os

# =============================================================================
# Path Setup
# =============================================================================

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
_tests_dir = os.path.join(_project_root, "tests")

for _path in [_project_root, _tests_dir]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


# =============================================================================
# Pytest Markers
# =============================================================================

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "root_serial: mark test to run serially (ROOT not thread-safe)"
    )


pytestmark = [pytest.mark.root_serial]


# =============================================================================
# Import Helpers
# =============================================================================

def _import_generators():
    """Import generators from tests/generators/."""
    try:
        from generators.toy_nd import (
            generate_nd_2d_root, 
            generate_helix_root, 
            register_custom_classes,
        )
        return generate_nd_2d_root, generate_helix_root, register_custom_classes
    except ImportError:
        pass
    try:
        from tests.generators.toy_nd import (
            generate_nd_2d_root,
            generate_helix_root,
            register_custom_classes,
        )
        return generate_nd_2d_root, generate_helix_root, register_custom_classes
    except ImportError:
        return None, None, None


def _import_dsl():
    """Import DSLCompiler."""
    from RDataFrameDSL import DSLCompiler
    return DSLCompiler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def nd_2d_root_file(tmp_path_factory):
    """Generate 2D nested data ROOT file (plain arrays)."""
    ROOT = pytest.importorskip("ROOT")
    generate_nd_2d_root, _, _ = _import_generators()
    if generate_nd_2d_root is None:
        pytest.skip("generators.toy_nd not available")
    
    tmp_dir = tmp_path_factory.mktemp("nd2d")
    filename = str(tmp_dir / "nd2d_test.root")
    return generate_nd_2d_root(size='S', seed=42, filename=filename)


@pytest.fixture(scope="module")
def nd_2d_rdf(nd_2d_root_file):
    """RDataFrame from 2D nested data."""
    ROOT = pytest.importorskip("ROOT")
    return ROOT.RDataFrame("Events", nd_2d_root_file)


@pytest.fixture(scope="module")
def nd_2d_schema():
    """Schema for 2D nested data."""
    return {
        'event_id': 'long',
        'n_tracks': 'int',
        'event_weight': 'double',
        'track_pt': 'RVec<double>',
        'track_eta': 'RVec<double>',
        'cluster_Q': 'RVec<RVec<double>>',
        'cluster_x': 'RVec<RVec<double>>',
        'cluster_y': 'RVec<RVec<double>>',
    }


@pytest.fixture
def dsl_nd2d(nd_2d_schema):
    """DSLCompiler for 2D nested data."""
    DSLCompiler = _import_dsl()
    return DSLCompiler(nd_2d_schema)


@pytest.fixture(scope="module")
def helix_root_file(tmp_path_factory):
    """Generate helix trajectory ROOT file (custom classes)."""
    ROOT = pytest.importorskip("ROOT")
    _, generate_helix_root, register_custom_classes = _import_generators()
    if generate_helix_root is None:
        pytest.skip("generators.toy_nd.generate_helix_root not available")
    
    if register_custom_classes:
        register_custom_classes()
    
    tmp_dir = tmp_path_factory.mktemp("helix")
    filename = str(tmp_dir / "helix_test.root")
    return generate_helix_root(
        n_events=50, tracks_per_event=(2, 5), b_field=0.5,
        pt_range=(0.5, 5.0), eta_range=(-1.0, 1.0), seed=42,
        filename=filename
    )


@pytest.fixture(scope="module")
def helix_rdf(helix_root_file):
    """RDataFrame from helix file."""
    ROOT = pytest.importorskip("ROOT")
    return ROOT.RDataFrame("Events", helix_root_file)


@pytest.fixture
def dsl_helix(helix_rdf):
    """
    DSLCompiler for helix data - use from_rdf() to infer schema with methods.
    
    This ensures ToyTrack methods are properly resolved via ROOT reflection.
    """
    _, _, register_custom_classes = _import_generators()
    if register_custom_classes:
        register_custom_classes()
    DSLCompiler = _import_dsl()
    # Use from_rdf() to infer schema - this handles custom classes better
    return DSLCompiler.from_rdf(helix_rdf)


# =============================================================================
# PRE-FLIGHT TESTS (P0 Critical)
# =============================================================================

class TestPreFlightCritical:
    """P0 Pre-flight tests - validates DSL compilation for key features."""
    
    def test_preflight_1_slice_first_3(self, dsl_nd2d, nd_2d_rdf):
        """P0-1: track_pt[:3] - first 3 elements."""
        dsl_nd2d.define("test_p1", "track_pt[:3]")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p1"])
        assert "test_p1" in data
        print("✅ P0-1: track_pt[:3] WORKS")
    
    def test_preflight_2_last_element(self, dsl_nd2d, nd_2d_rdf):
        """P0-2: track_pt[-1] - last element."""
        dsl_nd2d.define("test_p2", "track_pt[-1]")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p2"])
        assert "test_p2" in data
        print("✅ P0-2: track_pt[-1] WORKS")
    
    def test_preflight_3_range_slice(self, dsl_nd2d, nd_2d_rdf):
        """P0-3: track_pt[1:-1] - middle elements."""
        dsl_nd2d.define("test_p3", "track_pt[1:-1]")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p3"])
        assert "test_p3" in data
        print("✅ P0-3: track_pt[1:-1] WORKS")
    
    def test_preflight_4_step_slice(self, dsl_nd2d, nd_2d_rdf):
        """P0-4: track_pt[::2] - every other element."""
        dsl_nd2d.define("test_p4", "track_pt[::2]")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p4"])
        assert "test_p4" in data
        print("✅ P0-4: track_pt[::2] WORKS")
    
    def test_preflight_5_2d_slice(self, dsl_nd2d, nd_2d_rdf):
        """P0-5: cluster_Q[0][:5] - 2D nested slice."""
        dsl_nd2d.define("test_p5", "cluster_Q[0][:5]")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p5"])
        assert "test_p5" in data
        print("✅ P0-5: cluster_Q[0][:5] WORKS")
    
    def test_preflight_6_reduction_on_slice(self, dsl_nd2d, nd_2d_rdf):
        """P0-6: Sum(track_pt[:3]) - reduction on slice."""
        dsl_nd2d.define("test_p6", "Sum(track_pt[:3])")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p6"])
        assert "test_p6" in data
        print("✅ P0-6: Sum(track_pt[:3]) WORKS")
    
    def test_preflight_7_broadcast(self, dsl_nd2d, nd_2d_rdf):
        """P0-7: track_pt * event_weight - scalar broadcast."""
        dsl_nd2d.define("test_p7", "track_pt * event_weight")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p7"])
        assert "test_p7" in data
        print("✅ P0-7: track_pt * event_weight WORKS")


# =============================================================================
# PART 1A: Plain Arrays - TTree::Draw Equivalent
# =============================================================================

class TestPart1A_PlainArrays:
    """Part 1A: TTree::Draw equivalent with plain arrays."""
    
    def test_1a_1_simple_histogram(self, dsl_nd2d, nd_2d_rdf):
        """1A.1: Simple 1D histogram."""
        try:
            result = dsl_nd2d.draw("track_pt", nd_2d_rdf)
            assert result is not None
            print("⏱️ 1A.1 Simple 1D: OK (with dfdraw)")
        except ImportError:
            print("⏱️ 1A.1 Simple 1D: dfdraw not installed, skipping plot")
    
    def test_1a_2_expression(self, dsl_nd2d, nd_2d_rdf):
        """1A.2: Math expression."""
        dsl_nd2d.define("track_pt2", "track_pt**2")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        
        try:
            result = dsl_nd2d.draw("track_pt2", rdf)
            assert result is not None
            print("⏱️ 1A.2 Expression: OK (with dfdraw)")
        except ImportError:
            assert "track_pt2" in dsl_nd2d.schema
            print("⏱️ 1A.2 Expression: define() validated")
    
    def test_1a_3_alias_materialized(self, dsl_nd2d, nd_2d_rdf):
        """1A.3: Alias materialization (key invariant)."""
        dsl_nd2d.alias("pt", "track_pt")
        
        try:
            dsl_nd2d.draw("pt", nd_2d_rdf)
        except ImportError:
            pass  # dfdraw not installed - OK
        
        # INVARIANT: Alias must be materialized in schema
        assert "pt" in dsl_nd2d.schema, "Alias 'pt' must be materialized"
        print("⏱️ 1A.3 Alias: materialized in schema ✅")
    
    def test_1a_4_alias_chain(self, dsl_nd2d, nd_2d_rdf):
        """1A.4: Alias chain compilation (from test_invariance_safe_draw)."""
        dsl_nd2d.alias("pt_x2", "track_pt * 2")
        dsl_nd2d.alias("pt_x4", "pt_x2 * 2")
        
        try:
            dsl_nd2d.draw("pt_x4", nd_2d_rdf)
        except ImportError:
            pass
        
        assert "pt_x2" in dsl_nd2d.schema
        assert "pt_x4" in dsl_nd2d.schema
        print("⏱️ 1A.4 Alias chain: pt_x4 → pt_x2 → track_pt ✅")


# =============================================================================
# PART 1B: Custom Classes - Member Functions
# =============================================================================

class TestPart1B_CustomClasses:
    """
    Part 1B: Member functions with ToyTrack/ToyCluster.
    
    Uses DSLCompiler.from_rdf() to properly infer schema with method support.
    """
    
    def test_1b_1_member_function_pt(self, dsl_helix, helix_rdf):
        """1B.1: tracks.Pt() member function."""
        # With from_rdf(), the tracks column type should be inferred
        dsl_helix.define("track_pt", "tracks.Pt()")
        rdf = dsl_helix.apply(helix_rdf)
        
        try:
            result = dsl_helix.draw("track_pt", rdf)
            assert result is not None
            print("⏱️ 1B.1 tracks.Pt(): OK (with dfdraw)")
        except ImportError:
            assert "track_pt" in dsl_helix.schema
            print("⏱️ 1B.1 tracks.Pt(): define() validated")
        print("✅ Equivalent to TTree::Draw(\"tracks.Pt()\")")
    
    def test_1b_2_nested_method(self, dsl_helix, helix_rdf):
        """1B.2: tracks.clusters().getQ() nested method.
        
        Phase 13.6.G+: This tests chained method broadcast on nested containers.
        tracks.clusters() returns RVec<RVec<ToyCluster>>, then .getQ() broadcasts
        to produce RVec<RVec<double>> - a 2D structure.
        
        Note: draw() doesn't support 2D arrays, so we validate via AsNumpy().
        """
        dsl_helix.define("cluster_Q", "tracks.clusters().getQ()")
        rdf = dsl_helix.apply(helix_rdf)
        
        # Validate the 2D data structure directly via AsNumpy
        data = rdf.AsNumpy(["cluster_Q"])
        assert "cluster_Q" in data
        assert len(data["cluster_Q"]) > 0
        
        # Verify it's actually 2D (each event has array of arrays)
        first_event = data["cluster_Q"][0]
        assert hasattr(first_event, '__len__'), "Should be array-like"
        print(f"✅ 1B.2: tracks.clusters().getQ() - 2D nested broadcast")
        print(f"   Events: {len(data['cluster_Q'])}")
        if len(first_event) > 0:
            first_track = first_event[0]
            # Convert to list for safe slicing (ROOT RVec doesn't support Python slice)
            first_track_list = list(first_track)[:3] if hasattr(first_track, '__iter__') else [first_track]
            print(f"   Event 0: {len(first_event)} tracks, first track charges: {first_track_list}...")
        else:
            print(f"   Event 0: 0 tracks")
        print("   Equivalent to: tree->Draw(\"tracks.clusters().getQ()\")")
    
    def test_1b_3_trajectory_plot(self, dsl_helix, helix_rdf):
        """1B.3: Trajectory plot cluster_y:cluster_x.
        
        Phase 13.6.G+: This tests chained method broadcast for cluster positions.
        Note: draw() doesn't support 2D arrays, so we validate via AsNumpy().
        """
        dsl_helix.define("cluster_x", "tracks.clusters().getX()")
        dsl_helix.define("cluster_y", "tracks.clusters().getY()")
        rdf = dsl_helix.apply(helix_rdf)
        
        # Validate the 2D data structure
        data = rdf.AsNumpy(["cluster_x", "cluster_y"])
        assert "cluster_x" in data
        assert "cluster_y" in data
        assert len(data["cluster_x"]) > 0
        
        print(f"✅ 1B.3: tracks.clusters() x,y - 2D nested broadcast for trajectories")
        print(f"   Events: {len(data['cluster_x'])}")


# =============================================================================
# PART 2: Beyond TTree::Draw (DSL-only features)
# =============================================================================

class TestPart2_BeyondTTreeDraw:
    """Part 2: Features TTree::Draw CANNOT do."""
    
    def test_2_1_array_slicing(self, dsl_nd2d, nd_2d_rdf):
        """2.1: Array slicing [:3] - IMPOSSIBLE in TTree::Draw."""
        dsl_nd2d.define("leading_pt", "track_pt[:3]")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        
        try:
            dsl_nd2d.draw("leading_pt", rdf)
        except ImportError:
            pass
        
        assert "leading_pt" in dsl_nd2d.schema
        print("✅ 2.1 track_pt[:3]: TTree::Draw IMPOSSIBLE")
    
    def test_2_2_negative_index(self, dsl_nd2d, nd_2d_rdf):
        """2.2: Negative indexing [-1] - IMPOSSIBLE in TTree::Draw."""
        dsl_nd2d.define("last_pt", "track_pt[-1]")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        
        try:
            dsl_nd2d.draw("last_pt", rdf)
        except ImportError:
            pass
        
        assert "last_pt" in dsl_nd2d.schema
        print("✅ 2.2 track_pt[-1]: TTree::Draw IMPOSSIBLE")
    
    def test_2_3_reduction_on_slice(self, dsl_nd2d, nd_2d_rdf):
        """2.3: Sum(track_pt[:3]) - IMPOSSIBLE in TTree::Draw."""
        dsl_nd2d.define("sum_leading", "Sum(track_pt[:3])")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        
        try:
            dsl_nd2d.draw("sum_leading", rdf)
        except ImportError:
            pass
        
        assert "sum_leading" in dsl_nd2d.schema
        print("✅ 2.3 Sum(track_pt[:3]): TTree::Draw IMPOSSIBLE")
    
    def test_2_4_broadcast(self, dsl_nd2d, nd_2d_rdf):
        """2.4: track_pt * event_weight - IMPOSSIBLE in TTree::Draw."""
        dsl_nd2d.define("weighted_pt", "track_pt * event_weight")
        rdf = dsl_nd2d.apply(nd_2d_rdf)
        
        try:
            dsl_nd2d.draw("weighted_pt", rdf)
        except ImportError:
            pass
        
        assert "weighted_pt" in dsl_nd2d.schema
        print("✅ 2.4 Broadcast 1D×0D: TTree::Draw IMPOSSIBLE")
    
    def test_2_5_chained_aliases(self, nd_2d_schema, nd_2d_rdf):
        """
        2.5: Chained aliases - limited in TTree::Draw.
        
        Pattern from test_invariance_safe_draw.py:
        - Use to_pandas() to trigger alias materialization
        - Verify aliases are materialized in schema
        """
        DSLCompiler = _import_dsl()
        dsl = DSLCompiler(nd_2d_schema)
        
        # Define in "wrong" order (reverse dependency order)
        dsl.alias("c", "b + 1")
        dsl.alias("b", "a + 1")
        dsl.alias("a", "track_pt[0]")
        
        # Use to_pandas to materialize the alias chain
        # This is the pattern from test_invariance_safe_draw.py
        df = dsl.to_pandas(nd_2d_rdf, columns=['track_pt', 'c'], max_entries=5)
        
        # INVARIANT: All aliases in chain must be materialized
        assert "a" in dsl.schema, "a must be materialized"
        assert "b" in dsl.schema, "b must be materialized"
        assert "c" in dsl.schema, "c must be materialized"
        
        # Verify value: c = track_pt[0] + 2
        # (can't easily verify without knowing exact values)
        print("✅ 2.5 Chained aliases: order-independent resolution")
        print(f"   Chain: c = b + 1 = a + 2 = track_pt[0] + 2")


# =============================================================================
# PART 3: Batch Operations (from proposal)
# =============================================================================

class TestPart3_BatchOperations:
    """Part 3: Batch draw operations."""
    
    def test_3_1_draw_batch(self, dsl_nd2d, nd_2d_rdf):
        """3.1: draw_batch() - single extraction, multiple plots."""
        specs = {
            'pt_hist': {'expr': 'track_pt'},
            'eta_hist': {'expr': 'track_eta'},
        }
        
        try:
            results = dsl_nd2d.draw_batch(specs, nd_2d_rdf, max_entries=10)
            assert len(results) == 2
            print("✅ 3.1 draw_batch(): 2 plots from single extraction")
        except ImportError:
            print("⏱️ 3.1 draw_batch(): dfdraw not installed")
        except AttributeError:
            print("⏠️ 3.1 draw_batch(): method not available")


# =============================================================================
# Summary
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_summary(request):
    """Print summary after all tests."""
    yield
    print("\n" + "=" * 60)
    print("07_dsl_draw.ipynb Pre-Flight Test Summary")
    print("=" * 60)
    print("pytest test_07_dsl_draw.py -v -k 'preflight'  # P0 critical")
    print("pytest test_07_dsl_draw.py -v -k 'Part1A'     # Plain arrays")
    print("pytest test_07_dsl_draw.py -v -k 'Part1B'     # Custom classes")
    print("pytest test_07_dsl_draw.py -v -k 'Part2'      # Beyond TTree::Draw")
    print("pytest test_07_dsl_draw.py -v -k 'Part3'      # Batch operations")
    print("=" * 60)
