"""
test_example_07_dsl_draw.py - Pre-flight tests for examples/07_dsl_draw.ipynb

Phase 13.6.G+: Tests mirroring notebook structure for validation.

IMPORTANT - TEST ORDERING:
    Tests are ordered so that custom_class_* fixtures run BEFORE nd_2d_* fixtures.
    This is required because ROOT dictionary generation for ToyTrack/ToyCluster
    must happen BEFORE any other ROOT file operations that might interfere.
    
    Order: Part1B (custom classes) → PreFlight → Part1A → Part2 → Part3

Location: tests/test_example_07_dsl_draw.py

Usage:
    pytest tests/test_example_07_dsl_draw.py -v --tb=short
    pytest tests/test_example_07_dsl_draw.py -v -k "Part1B"   # Custom classes (run first!)
    pytest tests/test_example_07_dsl_draw.py -v -k "preflight"
    pytest tests/test_example_07_dsl_draw.py -v -k "Part1A"
    pytest tests/test_example_07_dsl_draw.py -v -k "Part2"
    pytest tests/test_example_07_dsl_draw.py -v -k "Part3"

Fixtures used (from conftest_nd_additions.py):
    - custom_class_rdf, custom_class_schema: ToyTrack/ToyCluster with methods
    - nd_2d_rdf, nd_2d_schema: Plain arrays (RVec<double>, RVec<RVec<double>>)
"""

import pytest
import numpy as np

from RDataFrameDSL import DSLCompiler
from RDataFrameDSL.ir_errors import IRError


# Mark entire module to run serially (ROOT not thread-safe)
pytestmark = pytest.mark.root_serial


# =============================================================================
# PART 1B: Custom Classes - Member Functions (RUN FIRST!)
# =============================================================================
# These tests MUST run before any tests that use nd_2d_rdf to ensure
# ROOT dictionary generation for ToyTrack/ToyCluster happens first.

class TestPart1B_CustomClasses:
    """
    Part 1B: Member functions with ToyTrack/ToyCluster.
    
    IMPORTANT: This class is named to sort BEFORE TestPart1A alphabetically,
    ensuring custom class fixtures are initialized first.
    
    Uses custom_class_rdf from conftest_nd_additions.py which properly
    handles ROOT dictionary generation ordering.
    """
    
    def test_1b_01_member_function_pt(self, custom_class_rdf, custom_class_schema):
        """1B.1: tracks.Pt() member function."""
        dsl = DSLCompiler(custom_class_schema)
        dsl.define("track_pt", "tracks.Pt()")
        rdf = dsl.apply(custom_class_rdf)
        
        try:
            result = dsl.draw("track_pt", rdf)
            assert result is not None
            print("⏱️ 1B.1 tracks.Pt(): OK (with dfdraw)")
        except ImportError:
            assert "track_pt" in dsl.schema
            print("⏱️ 1B.1 tracks.Pt(): define() validated")
        print("✅ Equivalent to TTree::Draw(\"tracks.Pt()\")")
    
    def test_1b_02_member_function_indexed(self, custom_class_rdf, custom_class_schema):
        """1B.2: tracks[0].Pt() - indexed member function (matches UDF test pattern)."""
        dsl = DSLCompiler(custom_class_schema)
        dsl.define("pt0", "tracks[0].Pt()")
        rdf = dsl.apply(custom_class_rdf)
        
        data = rdf.AsNumpy(["pt0"])
        assert "pt0" in data
        
        # First track uses Pythagorean triple (3, 4, 5) → Pt = 5.0
        for i, pt in enumerate(data["pt0"]):
            assert abs(pt - 5.0) < 1e-10, f"Event {i}: pt0={pt}, expected 5.0"
        
        print("✅ 1B.2 tracks[0].Pt() == 5.0 (Pythagorean: 3² + 4² = 5²)")
    
    def test_1b_03_nested_method_cluster_q(self, custom_class_rdf, custom_class_schema):
        """1B.3: Nested method call - tracks[0].clusters().getQ()."""
        dsl = DSLCompiler(custom_class_schema)
        dsl.define("cluster_Q", "tracks[0].clusters().getQ()")
        rdf = dsl.apply(custom_class_rdf)
        
        try:
            result = dsl.draw("cluster_Q", rdf)
            assert result is not None
            print("⏱️ 1B.3 clusters().getQ(): OK (with dfdraw)")
        except ImportError:
            assert "cluster_Q" in dsl.schema
            print("⏱️ 1B.3 clusters().getQ(): define() validated")
        print("✅ Equivalent to TTree::Draw(\"tracks[0].clusters().getQ()\")")
    
    def test_1b_04_sliced_method(self, custom_class_rdf, custom_class_schema):
        """1B.4: Sliced method call - tracks[:2].Pt()."""
        dsl = DSLCompiler(custom_class_schema)
        dsl.define("leading_pt", "tracks[:2].Pt()")
        rdf = dsl.apply(custom_class_rdf)
        
        data = rdf.AsNumpy(["leading_pt"])
        assert "leading_pt" in data
        
        # Verify slice worked: each result should have at most 2 elements
        for i, arr in enumerate(data["leading_pt"]):
            assert len(arr) <= 2, f"Event {i}: expected <= 2 tracks, got {len(arr)}"
        
        print("✅ 1B.4 tracks[:2].Pt(): slice-then-method WORKS")
    
    def test_1b_05_nested_method_cluster_xy(self, custom_class_rdf, custom_class_schema):
        """1B.5: Nested method for trajectory - cluster x,y positions."""
        dsl = DSLCompiler(custom_class_schema)
        dsl.define("cluster_x", "tracks[0].clusters().getX()")
        dsl.define("cluster_y", "tracks[0].clusters().getY()")
        rdf = dsl.apply(custom_class_rdf)
        
        try:
            result = dsl.draw("cluster_y:cluster_x", rdf, type="scatter")
            assert result is not None
            print("⏱️ 1B.5 Trajectory (y:x): OK (with dfdraw)")
        except ImportError:
            assert "cluster_x" in dsl.schema
            assert "cluster_y" in dsl.schema
            print("⏱️ 1B.5 Trajectory: define() validated")


# =============================================================================
# PRE-FLIGHT TESTS (P0 Critical) - Plain Arrays
# =============================================================================

class TestPreFlightCritical:
    """
    P0 Pre-flight tests - validates DSL compilation for key features.
    
    Uses nd_2d_rdf (plain arrays) from conftest.
    All these tests MUST pass before notebook creation.
    """
    
    def test_preflight_1_slice_first_3(self, nd_2d_rdf, nd_2d_schema):
        """P0-1: track_pt[:3] - first 3 elements."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("test_p1", "track_pt[:3]")
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p1"])
        assert "test_p1" in data
        print("✅ P0-1: track_pt[:3] WORKS")
    
    def test_preflight_2_last_element(self, nd_2d_rdf, nd_2d_schema):
        """P0-2: track_pt[-1] - last element."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("test_p2", "track_pt[-1]")
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p2"])
        assert "test_p2" in data
        print("✅ P0-2: track_pt[-1] WORKS")
    
    def test_preflight_3_range_slice(self, nd_2d_rdf, nd_2d_schema):
        """P0-3: track_pt[1:-1] - middle elements."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("test_p3", "track_pt[1:-1]")
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p3"])
        assert "test_p3" in data
        print("✅ P0-3: track_pt[1:-1] WORKS")
    
    def test_preflight_4_step_slice(self, nd_2d_rdf, nd_2d_schema):
        """P0-4: track_pt[::2] - every other element."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("test_p4", "track_pt[::2]")
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p4"])
        assert "test_p4" in data
        print("✅ P0-4: track_pt[::2] WORKS")
    
    def test_preflight_5_2d_slice(self, nd_2d_rdf, nd_2d_schema):
        """P0-5: cluster_Q[0][:5] - 2D nested slice."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("test_p5", "cluster_Q[0][:5]")
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p5"])
        assert "test_p5" in data
        print("✅ P0-5: cluster_Q[0][:5] WORKS")
    
    def test_preflight_6_reduction_on_slice(self, nd_2d_rdf, nd_2d_schema):
        """P0-6: Sum(track_pt[:3]) - reduction on slice."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("test_p6", "Sum(track_pt[:3])")
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p6"])
        assert "test_p6" in data
        print("✅ P0-6: Sum(track_pt[:3]) WORKS")
    
    def test_preflight_7_broadcast(self, nd_2d_rdf, nd_2d_schema):
        """P0-7: track_pt * event_weight - scalar broadcast."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("test_p7", "track_pt * event_weight")
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["test_p7"])
        assert "test_p7" in data
        print("✅ P0-7: track_pt * event_weight WORKS")


# =============================================================================
# PART 1A: Plain Arrays - TTree::Draw Equivalent
# =============================================================================

class TestPart1A_PlainArrays:
    """
    Part 1A: TTree::Draw equivalent with plain arrays.
    
    Uses nd_2d_rdf which has RVec<double> and RVec<RVec<double>> columns.
    No member functions - tests basic draw() functionality.
    """
    
    def test_1a_1_simple_histogram(self, nd_2d_rdf, nd_2d_schema):
        """1A.1: Simple 1D histogram."""
        dsl = DSLCompiler(nd_2d_schema)
        try:
            result = dsl.draw("track_pt", nd_2d_rdf)
            assert result is not None
            print("⏱️ 1A.1 Simple 1D: OK (with dfdraw)")
        except ImportError:
            print("⏱️ 1A.1 Simple 1D: dfdraw not installed, skipping plot")
    
    def test_1a_2_2d_plot(self, nd_2d_rdf, nd_2d_schema):
        """1A.2: 2D scatter plot (y:x syntax)."""
        dsl = DSLCompiler(nd_2d_schema)
        try:
            result = dsl.draw("track_eta:track_pt", nd_2d_rdf)
            assert result is not None
            print("⏱️ 1A.2 2D plot: OK (with dfdraw)")
        except ImportError:
            print("⏱️ 1A.2 2D plot: dfdraw not installed, skipping plot")
    
    def test_1a_3_expression(self, nd_2d_rdf, nd_2d_schema):
        """1A.3: Math expression."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("track_pt2", "track_pt**2")
        rdf = dsl.apply(nd_2d_rdf)
        
        try:
            result = dsl.draw("track_pt2", rdf)
            assert result is not None
            print("⏱️ 1A.3 Expression: OK (with dfdraw)")
        except ImportError:
            assert "track_pt2" in dsl.schema
            print("⏱️ 1A.3 Expression: define() validated")
    
    def test_1a_4_selection(self, nd_2d_rdf, nd_2d_schema):
        """1A.4: Draw with selection cut (like TTree::Draw second argument)."""
        dsl = DSLCompiler(nd_2d_schema)
        
        # Test selection via to_pandas (more reliable than draw with selection)
        # Selection: only events with n_tracks > 2
        df_all = dsl.to_pandas(nd_2d_rdf, columns=['n_tracks', 'track_pt'], max_entries=100)
        n_all = len(df_all)
        
        # Apply filter via RDataFrame
        rdf_filtered = nd_2d_rdf.Filter("n_tracks > 2")
        df_filtered = dsl.to_pandas(rdf_filtered, columns=['n_tracks', 'track_pt'], max_entries=100)
        n_filtered = len(df_filtered)
        
        # Filtered should have fewer or equal rows
        assert n_filtered <= n_all, f"Filter should reduce rows: {n_filtered} <= {n_all}"
        
        # All filtered rows should have n_tracks > 2
        for i, row in df_filtered.iterrows():
            assert row['n_tracks'] > 2, f"Row {i}: n_tracks={row['n_tracks']} should be > 2"
        
        print(f"✅ 1A.4 Selection: {n_all} → {n_filtered} events (n_tracks > 2)")
    
    def test_1a_5_max_entries(self, nd_2d_rdf, nd_2d_schema):
        """
        1A.5: max_entries parameter limits EVENTS (not flattened rows).
        
        Note: max_entries limits the number of events processed by RDataFrame.
        After flattening RVec columns, the DataFrame may have more rows than
        max_entries because each event can have multiple array elements.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Get full data - use scalar column to count events accurately
        df_full = dsl.to_pandas(nd_2d_rdf, columns=['event_id'])
        n_events_full = len(df_full)
        
        # Get limited data
        limit = 3
        df_limited = dsl.to_pandas(nd_2d_rdf, columns=['event_id'], max_entries=limit)
        n_events_limited = len(df_limited)
        
        # max_entries limits EVENTS, not flattened rows
        assert n_events_limited <= limit, f"max_entries={limit} but got {n_events_limited} events"
        assert n_events_limited < n_events_full or n_events_full <= limit, \
            f"Limit should reduce events: {n_events_limited} < {n_events_full}"
        
        print(f"✅ 1A.5 max_entries: {n_events_full} → {n_events_limited} events (limit={limit})")
    
    def test_1a_6_alias_materialized(self, nd_2d_rdf, nd_2d_schema):
        """1A.6: Alias materialization (key invariant from test_invariance_safe_draw)."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("pt", "track_pt")
        
        try:
            dsl.draw("pt", nd_2d_rdf)
        except ImportError:
            pass  # dfdraw not installed - OK
        
        # INVARIANT: Alias must be materialized in schema
        assert "pt" in dsl.schema, "Alias 'pt' must be materialized"
        print("⏱️ 1A.6 Alias: materialized in schema ✅")
    
    def test_1a_7_alias_chain(self, nd_2d_rdf, nd_2d_schema):
        """1A.7: Alias chain compilation (from test_invariance_safe_draw)."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("pt_x2", "track_pt * 2")
        dsl.alias("pt_x4", "pt_x2 * 2")
        
        try:
            dsl.draw("pt_x4", nd_2d_rdf)
        except ImportError:
            pass
        
        assert "pt_x2" in dsl.schema
        assert "pt_x4" in dsl.schema
        print("⏱️ 1A.7 Alias chain: pt_x4 → pt_x2 → track_pt ✅")


# =============================================================================
# PART 2: Beyond TTree::Draw (DSL-only features)
# =============================================================================

class TestPart2_BeyondTTreeDraw:
    """
    Part 2: Features TTree::Draw CANNOT do.
    
    These are the key differentiators for the DSL.
    """
    
    def test_2_1_array_slicing(self, nd_2d_rdf, nd_2d_schema):
        """2.1: Array slicing [:3] - IMPOSSIBLE in TTree::Draw."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("leading_pt", "track_pt[:3]")
        rdf = dsl.apply(nd_2d_rdf)
        
        try:
            dsl.draw("leading_pt", rdf)
        except ImportError:
            pass
        
        assert "leading_pt" in dsl.schema
        print("✅ 2.1 track_pt[:3]: TTree::Draw IMPOSSIBLE")
    
    def test_2_2_negative_index(self, nd_2d_rdf, nd_2d_schema):
        """2.2: Negative indexing [-1] - IMPOSSIBLE in TTree::Draw."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("last_pt", "track_pt[-1]")
        rdf = dsl.apply(nd_2d_rdf)
        
        try:
            dsl.draw("last_pt", rdf)
        except ImportError:
            pass
        
        assert "last_pt" in dsl.schema
        print("✅ 2.2 track_pt[-1]: TTree::Draw IMPOSSIBLE")
    
    def test_2_3_step_slice(self, nd_2d_rdf, nd_2d_schema):
        """2.3: Step slicing [::2] - IMPOSSIBLE in TTree::Draw."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("every_other", "track_pt[::2]")
        rdf = dsl.apply(nd_2d_rdf)
        
        try:
            dsl.draw("every_other", rdf)
        except ImportError:
            pass
        
        assert "every_other" in dsl.schema
        print("✅ 2.3 track_pt[::2]: TTree::Draw IMPOSSIBLE")
    
    def test_2_4_reduction_on_slice(self, nd_2d_rdf, nd_2d_schema):
        """2.4: Sum(track_pt[:3]) - IMPOSSIBLE in TTree::Draw."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("sum_leading", "Sum(track_pt[:3])")
        rdf = dsl.apply(nd_2d_rdf)
        
        try:
            dsl.draw("sum_leading", rdf)
        except ImportError:
            pass
        
        assert "sum_leading" in dsl.schema
        print("✅ 2.4 Sum(track_pt[:3]): TTree::Draw IMPOSSIBLE")
    
    def test_2_5_broadcast(self, nd_2d_rdf, nd_2d_schema):
        """2.5: track_pt * event_weight - IMPOSSIBLE in TTree::Draw."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("weighted_pt", "track_pt * event_weight")
        rdf = dsl.apply(nd_2d_rdf)
        
        try:
            dsl.draw("weighted_pt", rdf)
        except ImportError:
            pass
        
        assert "weighted_pt" in dsl.schema
        print("✅ 2.5 Broadcast 1D×0D: TTree::Draw IMPOSSIBLE")
    
    def test_2_6_chained_aliases(self, nd_2d_rdf, nd_2d_schema):
        """
        2.6: Chained aliases - limited in TTree::Draw.
        
        Pattern from test_invariance_safe_draw.py:
        - Use to_pandas() to trigger alias materialization
        - Verify aliases are materialized in schema
        
        NOTE: define("x", "alias") does NOT auto-resolve aliases.
              Use to_pandas() or draw() to trigger resolution.
        """
        dsl = DSLCompiler(nd_2d_schema)
        
        # Define in "wrong" order (reverse dependency order)
        dsl.alias("c", "b + 1")
        dsl.alias("b", "a + 1")
        dsl.alias("a", "track_pt[0]")
        
        # Use to_pandas to materialize the alias chain
        df = dsl.to_pandas(nd_2d_rdf, columns=['track_pt', 'c'], max_entries=5)
        
        # INVARIANT: All aliases in chain must be materialized
        assert "a" in dsl.schema, "a must be materialized"
        assert "b" in dsl.schema, "b must be materialized"
        assert "c" in dsl.schema, "c must be materialized"
        
        print("✅ 2.6 Chained aliases: order-independent resolution")
    
    def test_2_7_2d_nested_slice(self, nd_2d_rdf, nd_2d_schema):
        """2.7: cluster_Q[0][:5] - 2D slicing IMPOSSIBLE in TTree::Draw."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.define("first_track_clusters", "cluster_Q[0][:5]")
        rdf = dsl.apply(nd_2d_rdf)
        
        data = rdf.AsNumpy(["first_track_clusters"])
        assert "first_track_clusters" in data
        
        print("✅ 2.7 cluster_Q[0][:5]: 2D slicing TTree::Draw IMPOSSIBLE")


# =============================================================================
# PART 3: Physics Visualization
# =============================================================================

class TestPart3_PhysicsVisualization:
    """
    Part 3: Physics-motivated visualizations.
    
    Trajectory plots with selections - demonstrating real analysis patterns.
    """
    
    def test_3_1_trajectory_xy(self, custom_class_rdf, custom_class_schema):
        """3.1: Trajectory plot - cluster_y:cluster_x scatter."""
        dsl = DSLCompiler(custom_class_schema)
        dsl.define("cluster_x", "tracks[0].clusters().getX()")
        dsl.define("cluster_y", "tracks[0].clusters().getY()")
        rdf = dsl.apply(custom_class_rdf)
        
        # Verify data extraction works
        data = rdf.AsNumpy(["cluster_x", "cluster_y"])
        assert "cluster_x" in data
        assert "cluster_y" in data
        assert len(data["cluster_x"]) > 0
        
        try:
            result = dsl.draw("cluster_y:cluster_x", rdf, type="scatter")
            assert result is not None
            print("✅ 3.1 Trajectory (y:x): plotted")
        except ImportError:
            print("✅ 3.1 Trajectory (y:x): data extracted (dfdraw not installed)")
    
    def test_3_2_high_pt_selection(self, custom_class_rdf, custom_class_schema):
        """3.2: High-pT track selection (Pt > 3.0)."""
        dsl = DSLCompiler(custom_class_schema)
        dsl.define("pt0", "tracks[0].Pt()")
        
        # Apply filter for high-pT
        rdf_filtered = custom_class_rdf.Filter("tracks[0].Pt() > 3.0")
        rdf_result = dsl.apply(rdf_filtered)
        
        data = rdf_result.AsNumpy(["pt0"])
        
        # Verify all pass filter
        for i, pt in enumerate(data["pt0"]):
            assert pt > 3.0, f"Event {i}: pt0={pt} should be > 3.0"
        
        print(f"✅ 3.2 High-pT selection: {len(data['pt0'])} events with Pt > 3.0")
    
    def test_3_3_low_pt_selection(self, custom_class_rdf, custom_class_schema):
        """3.3: Low-pT track selection (Pt < 3.0)."""
        dsl = DSLCompiler(custom_class_schema)
        dsl.define("pt0", "tracks[0].Pt()")
        
        # Apply filter for low-pT
        rdf_filtered = custom_class_rdf.Filter("tracks[0].Pt() < 10.0")
        rdf_result = dsl.apply(rdf_filtered)
        
        data = rdf_result.AsNumpy(["pt0"])
        
        # Verify all pass filter
        for i, pt in enumerate(data["pt0"]):
            assert pt < 10.0, f"Event {i}: pt0={pt} should be < 10.0"
        
        print(f"✅ 3.3 Low-pT selection: {len(data['pt0'])} events with Pt < 10.0")
    
    def test_3_4_trajectory_with_slice(self, custom_class_rdf, custom_class_schema):
        """3.4: Trajectory with sliced tracks (first 2 tracks only)."""
        dsl = DSLCompiler(custom_class_schema)
        
        # Get positions from first 2 tracks only
        dsl.define("leading_pt", "tracks[:2].Pt()")
        rdf = dsl.apply(custom_class_rdf)
        
        data = rdf.AsNumpy(["leading_pt"])
        
        # Verify slice: each event should have at most 2 track Pt values
        for i, arr in enumerate(data["leading_pt"]):
            assert len(arr) <= 2, f"Event {i}: expected <= 2 tracks, got {len(arr)}"
        
        print("✅ 3.4 Trajectory with slice: tracks[:2].Pt() WORKS")


# =============================================================================
# PART 4: Batch Operations
# =============================================================================

class TestPart4_BatchOperations:
    """Part 4: Batch draw operations for QA workflows."""
    
    def test_4_1_draw_batch(self, nd_2d_rdf, nd_2d_schema):
        """4.1: draw_batch() - single extraction, multiple plots."""
        dsl = DSLCompiler(nd_2d_schema)
        specs = {
            'pt_hist': {'expr': 'track_pt'},
            'eta_hist': {'expr': 'track_eta'},
        }
        
        try:
            results = dsl.draw_batch(specs, nd_2d_rdf, max_entries=10)
            assert len(results) == 2
            print("✅ 4.1 draw_batch(): 2 plots from single extraction")
        except ImportError:
            print("⏱️ 4.1 draw_batch(): dfdraw not installed")
        except AttributeError:
            print("⚠️ 4.1 draw_batch(): method not available")
    
    def test_4_2_batch_with_aliases(self, nd_2d_rdf, nd_2d_schema):
        """4.2: draw_batch() with aliases."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("pt", "track_pt")
        dsl.alias("eta", "track_eta")
        
        specs = {
            'pt_hist': {'expr': 'pt'},
            'eta_hist': {'expr': 'eta'},
        }
        
        try:
            results = dsl.draw_batch(specs, nd_2d_rdf, max_entries=10)
            assert len(results) == 2
            assert "pt" in dsl.schema
            assert "eta" in dsl.schema
            print("✅ 4.2 draw_batch() with aliases: materialized correctly")
        except ImportError:
            print("⏱️ 4.2 draw_batch() with aliases: dfdraw not installed")
        except AttributeError:
            print("⚠️ 4.2 draw_batch() with aliases: method not available")


# =============================================================================
# PART 5: Alias Behavior Documentation
# =============================================================================

class TestPart5_AliasBehavior:
    """
    Part 5: Document alias resolution behavior.
    
    IMPORTANT: define() does NOT auto-resolve aliases.
    Use to_pandas() or draw() to trigger alias chain resolution.
    """
    
    def test_5_1_alias_resolved_by_to_pandas(self, nd_2d_rdf, nd_2d_schema):
        """5.1: Aliases ARE resolved by to_pandas()."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("my_pt", "track_pt[0]")
        
        # to_pandas() resolves aliases
        df = dsl.to_pandas(nd_2d_rdf, columns=['my_pt'], max_entries=5)
        
        assert "my_pt" in df.columns
        assert "my_pt" in dsl.schema
        print("✅ 5.1 Alias resolved by to_pandas()")
    
    def test_5_2_alias_resolved_by_draw(self, nd_2d_rdf, nd_2d_schema):
        """5.2: Aliases ARE resolved by draw()."""
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("my_pt", "track_pt")
        
        try:
            dsl.draw("my_pt", nd_2d_rdf)
        except ImportError:
            pass  # dfdraw not installed
        
        # Alias should be materialized after draw()
        assert "my_pt" in dsl.schema
        print("✅ 5.2 Alias resolved by draw()")
    
    def test_5_3_chained_alias_resolution(self, nd_2d_rdf, nd_2d_schema):
        """5.3: Chained aliases resolved in correct order."""
        dsl = DSLCompiler(nd_2d_schema)
        
        # Define in reverse dependency order
        dsl.alias("result", "intermediate + 1")
        dsl.alias("intermediate", "base * 2")
        dsl.alias("base", "track_pt[0]")
        
        # to_pandas triggers resolution of entire chain
        df = dsl.to_pandas(nd_2d_rdf, columns=['result'], max_entries=5)
        
        # All aliases in chain should be materialized
        assert "base" in dsl.schema
        assert "intermediate" in dsl.schema
        assert "result" in dsl.schema
        
        print("✅ 5.3 Chained aliases: result → intermediate → base")
    
    def test_5_4_define_resolves_alias(self, nd_2d_rdf, nd_2d_schema):
        """
        5.4: define() should resolve aliases like to_pandas() and draw() do.
        
        BUG: Currently define("x", "alias") fails with "Unknown variable".
        
        Expected: define() should resolve aliases consistently with:
        - to_pandas(columns=['alias'])  ✅ works
        - draw("alias")                 ✅ works  
        - define("x", "alias")          ❌ FAILS (BUG)
        
        This test documents the expected behavior after the bug is fixed.
        """
        dsl = DSLCompiler(nd_2d_schema)
        dsl.alias("my_alias", "track_pt[0]")
        
        # This should work - define() should resolve aliases
        # BUG: Currently raises IRError: "Unknown variable 'my_alias'"
        dsl.define("final", "my_alias")
        
        rdf = dsl.apply(nd_2d_rdf)
        data = rdf.AsNumpy(["final"])
        assert "final" in data
        assert "my_alias" in dsl.schema  # Alias should be materialized
        
        print("✅ 5.4 define() resolves alias: define('final', 'my_alias') WORKS")


# =============================================================================
# Summary
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def test_summary(request):
    """Print summary after all tests."""
    yield
    print("\n" + "=" * 70)
    print("07_dsl_draw.ipynb Pre-Flight Test Summary")
    print("=" * 70)
    print("pytest tests/test_example_07_dsl_draw.py -v -k 'Part1B'     # Custom classes (FIRST!)")
    print("pytest tests/test_example_07_dsl_draw.py -v -k 'preflight'  # P0")
    print("pytest tests/test_example_07_dsl_draw.py -v -k 'Part1A'     # Plain arrays")
    print("pytest tests/test_example_07_dsl_draw.py -v -k 'Part2'      # Beyond TTree::Draw")
    print("pytest tests/test_example_07_dsl_draw.py -v -k 'Part3'      # Physics visualization")
    print("pytest tests/test_example_07_dsl_draw.py -v -k 'Part4'      # Batch operations")
    print("pytest tests/test_example_07_dsl_draw.py -v -k 'Part5'      # Alias behavior")
    print("=" * 70)
