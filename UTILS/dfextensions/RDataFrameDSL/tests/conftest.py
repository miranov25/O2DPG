"""
pytest configuration for RDataFrameDSL tests.

Phase 12: Adds synthetic data fixtures for draw integration testing.
"""

import pytest
import numpy as np


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "future: marks tests for future phase techniques (Phase 7+)"
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
