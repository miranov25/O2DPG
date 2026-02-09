"""
Phase 13.7.B Integration & Invariance Tests: Parent-Child Indexing with Real Data.

These tests require:
- ROOT (PyROOT)
- libIndexHelpers.so + index_helpers.h (in ParentChildIndexing/)
- o2residuals_tpc.root (real ALICE TPC calibration data)

Run: pytest tests/test_index_helpers_integration.py -v -s

Spec reference: PHASE_13_7_B_v1.2_Specification.md §7.2, §7.3
"""

import os
import pytest
import numpy as np

# =============================================================================
# Skip if ROOT not available
# =============================================================================

try:
    import ROOT
    ROOT_AVAILABLE = True
except ImportError:
    ROOT_AVAILABLE = False

pytestmark = [
    pytest.mark.root_serial,
    pytest.mark.skipif(not ROOT_AVAILABLE, reason="ROOT not available"),
]

# =============================================================================
# Path discovery (test-only, no hardwired paths in core lib)
# =============================================================================

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)  # RDataFrameDSL/

# Search for ParentChildIndexing/ directory (contains .so, .h, .root)
_LIB_SEARCH_PATHS = [
    os.environ.get("INDEX_HELPERS_DIR", ""),
    os.path.join(_PROJECT_ROOT, "ParentChildIndexing"),
    os.path.join(os.getcwd(), "ParentChildIndexing"),
    "ParentChildIndexing",
]

# Search for data file
_DATA_SEARCH_PATHS = [
    os.environ.get("O2_RESIDUALS_FILE", ""),
    os.path.join(_PROJECT_ROOT, "ParentChildIndexing", "o2residuals_tpc.root"),
    os.path.join(os.getcwd(), "ParentChildIndexing", "o2residuals_tpc.root"),
    os.path.join(_PROJECT_ROOT, "o2residuals_tpc.root"),
    "o2residuals_tpc.root",
]


def _find_lib_dir():
    """Find directory containing libIndexHelpers.so."""
    for path in _LIB_SEARCH_PATHS:
        if path and os.path.isdir(path):
            if os.path.exists(os.path.join(path, "libIndexHelpers.so")):
                return os.path.abspath(path)
    return None


def _find_data_file():
    for path in _DATA_SEARCH_PATHS:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return None


RESIDUALS_FILE = _find_data_file()
INDEX_HELPERS_DIR = _find_lib_dir()
N_EVENTS = 5


def _skip_if_missing():
    if RESIDUALS_FILE is None:
        pytest.skip("o2residuals_tpc.root not found. "
                     "Set O2_RESIDUALS_FILE or place in ParentChildIndexing/")
    if INDEX_HELPERS_DIR is None:
        pytest.skip("libIndexHelpers.so not found. "
                     "Set INDEX_HELPERS_DIR or build in ParentChildIndexing/")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def index_helpers_loaded():
    """
    Set up ROOT's search paths so ensure_index_helpers_loaded() works.

    This adds ParentChildIndexing/ to ROOT's dynamic library path and
    include path — the same effect as running from that directory or
    having it on LD_LIBRARY_PATH.
    """
    _skip_if_missing()

    # Tell ROOT where to find .so and .h
    ROOT.gSystem.AddDynamicPath(INDEX_HELPERS_DIR)
    ROOT.gInterpreter.AddIncludePath(INDEX_HELPERS_DIR)

    # Now the core lib's ensure_index_helpers_loaded() will find them
    from RDataFrameDSL.index_helpers import ensure_index_helpers_loaded, reset_helpers_loaded
    reset_helpers_loaded()
    result = ensure_index_helpers_loaded()
    assert result, (
        f"ensure_index_helpers_loaded() failed after adding "
        f"{INDEX_HELPERS_DIR} to ROOT paths"
    )
    return True


@pytest.fixture(scope="module")
def rdf_with_friends(index_helpers_loaded):
    """Create RDataFrame with friend trees, matching explore_residuals_v3.py."""
    chain_resid = ROOT.TChain("unbinnedResid")
    chain_resid.Add(RESIDUALS_FILE)

    chain_tracks = ROOT.TChain("trackData")
    chain_tracks.Add(RESIDUALS_FILE)

    chain_resid.AddFriend(chain_tracks, "td")

    rdf = ROOT.RDataFrame(chain_resid).Range(N_EVENTS)
    return rdf


@pytest.fixture(scope="module")
def v3_reference_data(rdf_with_friends):
    """
    Reference data from exploration v3 approach (manual Define + AsNumpy).
    Ground truth for validating DSL integration.
    """
    rdf = rdf_with_friends

    rdf_expanded = rdf.Define("parentIdx",
        "RDataFrameDSL::IndexHelpers::ExpandParentIndexFromOffsets("
        "trackInfo.idxFirstResidual, (int)res.dy.size())")

    rdf_expanded = rdf_expanded.Define("res_dEdxTPC",
        "RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets<float>("
        "td.trk.dEdxTPC, trackInfo.idxFirstResidual, (int)res.dy.size())")

    rdf_expanded = rdf_expanded.Define("res_chi2TPC",
        "RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets<float>("
        "td.trk.chi2TPC, trackInfo.idxFirstResidual, (int)res.dy.size())")

    result = rdf_expanded.AsNumpy([
        "res.dy", "res.dz", "parentIdx",
        "res_dEdxTPC", "res_chi2TPC",
        "trackInfo.idxFirstResidual", "td.trk.dEdxTPC",
    ])

    return result


@pytest.fixture(scope="module")
def dsl_result(index_helpers_loaded):
    """
    Result from the DSL integration path:
    register_parent_child() → to_pandas()
    """
    from RDataFrameDSL import DSLCompiler

    chain_resid = ROOT.TChain("unbinnedResid")
    chain_resid.Add(RESIDUALS_FILE)
    chain_tracks = ROOT.TChain("trackData")
    chain_tracks.Add(RESIDUALS_FILE)
    chain_resid.AddFriend(chain_tracks, "td")
    rdf = ROOT.RDataFrame(chain_resid).Range(N_EVENTS)

    schema = {
        "res.dy": "RVec<short>",
        "res.dz": "RVec<short>",
        "td.trk.dEdxTPC": "RVec<float>",
        "td.trk.chi2TPC": "RVec<float>",
        "trackInfo.idxFirstResidual": "RVec<int>",
    }
    dsl = DSLCompiler(schema)

    dsl.register_parent_child(
        parent="td.trk",
        child="res",
        offset_column="trackInfo.idxFirstResidual",
    )

    df = dsl.to_pandas(rdf,
                       columns=["res.dy", "res.dz",
                                "td.trk.dEdxTPC", "td.trk.chi2TPC"],
                       parent_id_column=None)
    return df


# =============================================================================
# IT1: to_pandas() produces correct DataFrame
# =============================================================================

class TestIntegration:

    @pytest.mark.p0
    def test_IT1_dsl_produces_dataframe(self, dsl_result):
        """IT1: to_pandas with parent-child registration produces DataFrame."""
        df = dsl_result
        assert df is not None
        assert len(df) > 0
        assert "res.dy" in df.columns
        assert "res.dz" in df.columns
        assert "td.trk.dEdxTPC" in df.columns
        assert "td.trk.chi2TPC" in df.columns
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

    @pytest.mark.p0
    def test_IT1b_correct_row_count(self, dsl_result, v3_reference_data):
        """IT1b: Row count matches manual expansion (v3 reference)."""
        ref = v3_reference_data
        total_residuals_ref = sum(len(ref["res.dy"][i])
                                  for i in range(len(ref["res.dy"])))
        assert len(dsl_result) == total_residuals_ref, (
            f"DSL produced {len(dsl_result)} rows, "
            f"expected {total_residuals_ref} from v3 reference"
        )
        print(f"  Row count: {len(dsl_result)} (matches reference)")

    @pytest.mark.p0
    def test_IT2_expanded_dEdx_matches_reference(self, dsl_result,
                                                  v3_reference_data):
        """IT2: Expanded dEdxTPC values match v3 manual expansion."""
        ref = v3_reference_data

        ref_dEdx_flat = np.concatenate([
            np.array(ref["res_dEdxTPC"][i], dtype=np.float32)
            for i in range(len(ref["res_dEdxTPC"]))
        ])

        dsl_dEdx = dsl_result["td.trk.dEdxTPC"].values.astype(np.float32)

        np.testing.assert_allclose(
            dsl_dEdx, ref_dEdx_flat,
            rtol=1e-5,
            err_msg="DSL expanded dEdxTPC differs from v3 reference"
        )
        print(f"  dEdxTPC: {len(dsl_dEdx)} values match reference")

    @pytest.mark.p0
    def test_IT2b_expanded_chi2_matches_reference(self, dsl_result,
                                                   v3_reference_data):
        """IT2b: Expanded chi2TPC values match v3 manual expansion."""
        ref = v3_reference_data

        ref_chi2_flat = np.concatenate([
            np.array(ref["res_chi2TPC"][i], dtype=np.float32)
            for i in range(len(ref["res_chi2TPC"]))
        ])

        dsl_chi2 = dsl_result["td.trk.chi2TPC"].values.astype(np.float32)

        np.testing.assert_allclose(
            dsl_chi2, ref_chi2_flat,
            rtol=1e-5,
            err_msg="DSL expanded chi2TPC differs from v3 reference"
        )
        print(f"  chi2TPC: {len(dsl_chi2)} values match reference")

    @pytest.mark.p0
    def test_IT2c_child_columns_unchanged(self, dsl_result,
                                           v3_reference_data):
        """IT2c: Child columns (res.dy, res.dz) are unchanged."""
        ref = v3_reference_data

        ref_dy_flat = np.concatenate([
            np.array(ref["res.dy"][i], dtype=np.float64)
            for i in range(len(ref["res.dy"]))
        ])

        dsl_dy = dsl_result["res.dy"].values.astype(np.float64)

        np.testing.assert_array_equal(
            dsl_dy, ref_dy_flat,
            err_msg="res.dy values differ from reference"
        )
        print(f"  res.dy: {len(dsl_dy)} values match reference")


# =============================================================================
# INV1-INV3: Invariance Tests
# =============================================================================

class TestInvariance:

    @pytest.mark.p0
    def test_INV1_row_count_equals_child_size(self, v3_reference_data):
        """
        INV1: Row count equals total child elements across events.
        """
        ref = v3_reference_data
        total_children = sum(len(ref["res.dy"][i])
                             for i in range(len(ref["res.dy"])))

        total_parent_idx = sum(len(ref["parentIdx"][i])
                               for i in range(len(ref["parentIdx"])))

        assert total_children == total_parent_idx, (
            f"Mismatch: {total_children} children vs "
            f"{total_parent_idx} parent indices"
        )
        print(f"  INV1: {total_children} children == {total_parent_idx} "
              f"parent indices ✓")

    @pytest.mark.p0
    def test_INV2_expanded_values_match_direct_lookup(self,
                                                       v3_reference_data):
        """
        INV2: Expanded parent value for child i equals parent[parentIdx[i]].
        """
        ref = v3_reference_data

        for evt in range(len(ref["res.dy"])):
            parent_idx = np.array(ref["parentIdx"][evt], dtype=np.int32)
            dEdx_parent = np.array(ref["td.trk.dEdxTPC"][evt],
                                   dtype=np.float32)
            dEdx_expanded = np.array(ref["res_dEdxTPC"][evt],
                                     dtype=np.float32)

            dEdx_expected = dEdx_parent[parent_idx]

            np.testing.assert_allclose(
                dEdx_expanded, dEdx_expected,
                rtol=1e-5,
                err_msg=f"Event {evt}: expanded dEdx != parent[parentIdx]"
            )

        print(f"  INV2: expanded[i] == parent[parentIdx[i]] for all events ✓")

    @pytest.mark.p0
    def test_INV3_sum_expanded_equals_weighted_sum(self, v3_reference_data):
        """
        INV3: Sum of expanded values = sum over parents of (value × child_count).
        """
        ref = v3_reference_data

        for evt in range(len(ref["res.dy"])):
            offsets = np.array(ref["trackInfo.idxFirstResidual"][evt],
                               dtype=np.int32)
            dEdx_parent = np.array(ref["td.trk.dEdxTPC"][evt],
                                   dtype=np.float64)
            dEdx_expanded = np.array(ref["res_dEdxTPC"][evt],
                                     dtype=np.float64)
            n_children = len(ref["res.dy"][evt])

            child_counts = np.diff(offsets, append=n_children)

            weighted_sum = np.sum(dEdx_parent * child_counts)
            expanded_sum = np.sum(dEdx_expanded)

            np.testing.assert_allclose(
                expanded_sum, weighted_sum,
                rtol=1e-4,
                err_msg=(
                    f"Event {evt}: sum(expanded)={expanded_sum:.4f} != "
                    f"sum(parent*count)={weighted_sum:.4f}"
                )
            )

        print(f"  INV3: sum(expanded) == sum(value × child_count) "
              f"for all events ✓")

    @pytest.mark.p1
    def test_INV_expanded_constant_within_track(self, v3_reference_data):
        """
        Additional: expanded parent value is constant within each track.
        """
        ref = v3_reference_data

        for evt in range(len(ref["res.dy"])):
            parent_idx = np.array(ref["parentIdx"][evt], dtype=np.int32)
            dEdx_expanded = np.array(ref["res_dEdxTPC"][evt],
                                     dtype=np.float32)

            unique_parents = np.unique(parent_idx)
            for pid in unique_parents:
                mask = parent_idx == pid
                values = dEdx_expanded[mask]
                assert np.all(values == values[0]), (
                    f"Event {evt}, track {pid}: expanded dEdx not constant. "
                    f"Got {np.unique(values)}"
                )

        print(f"  INV_constant: expanded value constant within each track ✓")


# =============================================================================
# Edge case: child-only columns — no expansion
# =============================================================================

class TestNoExpansion:

    @pytest.mark.p1
    def test_child_only_no_expansion(self, index_helpers_loaded):
        """Child-only request works without expansion."""
        from RDataFrameDSL import DSLCompiler

        chain = ROOT.TChain("unbinnedResid")
        chain.Add(RESIDUALS_FILE)
        rdf = ROOT.RDataFrame(chain).Range(N_EVENTS)

        schema = {"res.dy": "RVec<short>", "res.dz": "RVec<short>"}
        dsl = DSLCompiler(schema)
        dsl.register_parent_child(
            parent="td.trk", child="res",
            offset_column="trackInfo.idxFirstResidual",
        )

        df = dsl.to_pandas(rdf, columns=["res.dy", "res.dz"],
                           parent_id_column=None)
        assert len(df) > 0
        assert "res.dy" in df.columns
        assert "res.dz" in df.columns
        print(f"  Child-only: {df.shape} — no expansion triggered ✓")
