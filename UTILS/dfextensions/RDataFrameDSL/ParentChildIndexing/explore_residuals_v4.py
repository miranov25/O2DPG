#!/usr/bin/env python3
"""
Phase 13.7.B Exploration v4: Cross-Level Queries + Snapshot + Invariance
=========================================================================

The query TTree::Draw can NOT do:
    qMaxTPC : td.trk.dEdxTPC
because qMaxTPC is per-cluster (~140 per track) and dEdxTPC is per-track.

This script:
  E0: Load libraries (O2 dict FIRST, then IndexHelpers)
  E1: Show the problem — different array lengths
  E2: Define expanded + extracted columns
  E3: Snapshot with Define'd sub-branches (avoids raw O2 objects)
  E4: Readback from snapshot — cross-level queries
  E5: DSL path — register_parent_child → to_pandas
  E6: Invariance tests (snapshot vs DSL vs manual)
  E7: Summary

Run from ParentChildIndexing/ directory:
  python -u explore_residuals_v4.py 2>&1 | tee explore_residuals_v4.log

Prerequisites:
  - O2 environment (alienv enter O2Physics/latest)
  - make -f Makefile.o2helpers   (builds libO2ResidualHelpers.so + pcm)
  - make                         (builds libIndexHelpers.so)
  - o2residuals_tpc.root
"""

import sys
import os
import numpy as np

print(f"Python: {sys.version}")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESIDUALS_FILE = "o2residuals_tpc.root"
SNAPSHOT_FILE = "snapshot_v4.root"
N_EVENTS = 5

# ============================================================
# E0: Load Libraries — ORDER MATTERS
# ============================================================
print("=" * 60)
print("E0: Load Libraries")
print("=" * 60)

import ROOT
print(f"  ROOT: {ROOT.gROOT.GetVersion()}")

# 1. O2 dictionary FIRST (provides RVec<o2::tpc::UnbinnedResid> etc.)
r0 = ROOT.gSystem.Load("./libO2ResidualHelpers.so")
assert r0 >= 0, f"Failed to load libO2ResidualHelpers.so (result={r0})"
ROOT.gInterpreter.Declare('#include "o2_residual_helpers.h"')
print(f"  libO2ResidualHelpers.so: loaded (result={r0})")

# 2. IndexHelpers (for ExpandToChildrenFromOffsets etc.)
r1 = ROOT.gSystem.Load("./libIndexHelpers.so")
assert r1 >= 0, f"Failed to load libIndexHelpers.so (result={r1})"
ROOT.gInterpreter.Declare('#include "index_helpers.h"')
print(f"  libIndexHelpers.so: loaded (result={r1})")

# 3. Verify symbols
_ = ROOT.RDataFrameDSL.IndexHelpers.ExpandParentIndexFromOffsets
_ = ROOT.RDataFrameDSL.O2ResidualHelpers.ExtractQMaxTPC
print(f"  Symbols verified: ExpandParentIndexFromOffsets, ExtractQMaxTPC")

print(f"\n[OK] E0 PASSED")

# ============================================================
# E1: The Problem — Different Array Lengths
# ============================================================
print("\n" + "=" * 60)
print("E1: The Problem — TTree::Draw can't do qMaxTPC:td.trk.dEdxTPC")
print("=" * 60)

chain = ROOT.TChain("unbinnedResid")
chain.Add(RESIDUALS_FILE)
chain_td = ROOT.TChain("trackData")
chain_td.Add(RESIDUALS_FILE)
chain.AddFriend(chain_td, "td")
rdf_probe = ROOT.RDataFrame(chain).Range(N_EVENTS)

probe = rdf_probe.AsNumpy(["res.dy", "td.trk.dEdxTPC",
                            "trackInfo.idxFirstResidual"])

print(f"\n  Per-event array lengths:")
print(f"  {'Event':>5}  {'res/cluster (child)':>20}  {'td.trk (parent)':>18}  {'ratio':>6}")
print(f"  {'─'*5}  {'─'*20}  {'─'*18}  {'─'*6}")
total_children = 0
total_parents = 0
for i in range(N_EVENTS):
    nc = len(probe["res.dy"][i])
    np_ = len(probe["td.trk.dEdxTPC"][i])
    total_children += nc
    total_parents += np_
    print(f"  {i:>5}  {nc:>20}  {np_:>18}  {nc/np_:>6.0f}:1")

print(f"\n  Total: {total_children} clusters/residuals, {total_parents} tracks")
print(f"  TTree::Draw(\"qMaxTPC:td.trk.dEdxTPC\") → FAILS (size mismatch)")

print(f"\n[OK] E1 PASSED")

# ============================================================
# E2: Define Expanded + Extracted Columns
# ============================================================
print("\n" + "=" * 60)
print("E2: Define expanded track + extracted cluster columns")
print("=" * 60)

# Fresh RDF for Snapshot
chain2 = ROOT.TChain("unbinnedResid")
chain2.Add(RESIDUALS_FILE)
chain_td2 = ROOT.TChain("trackData")
chain_td2.Add(RESIDUALS_FILE)
chain2.AddFriend(chain_td2, "td")
rdf2 = ROOT.RDataFrame(chain2).Range(N_EVENTS)

# --- Child-level: copy sub-branches as standalone (avoids raw O2 object) ---
rdf2 = rdf2.Define("dy",  "res.dy")
rdf2 = rdf2.Define("dz",  "res.dz")
rdf2 = rdf2.Define("row", "res.row")
rdf2 = rdf2.Define("sec", "res.sec")
print("  Copied: dy, dz, row, sec (from res.*)")

# --- Child-level: extract cluster charge via O2 interface ---
rdf2 = rdf2.Define("qMaxTPC",
    "RDataFrameDSL::O2ResidualHelpers::ExtractQMaxTPC(detInfo)")
rdf2 = rdf2.Define("qTotTPC",
    "RDataFrameDSL::O2ResidualHelpers::ExtractQTotTPC(detInfo)")
print("  Extracted: qMaxTPC, qTotTPC (from detInfo)")

# --- Parent index ---
rdf2 = rdf2.Define("parentIdx",
    "RDataFrameDSL::IndexHelpers::ExpandParentIndexFromOffsets("
    "trackInfo.idxFirstResidual, (int)res.dy.size())")
print("  Defined: parentIdx")

# --- Expanded parent→child ---
rdf2 = rdf2.Define("exp_dEdxTPC",
    "RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets<float>("
    "td.trk.dEdxTPC, trackInfo.idxFirstResidual, (int)res.dy.size())")
rdf2 = rdf2.Define("exp_chi2TPC",
    "RDataFrameDSL::IndexHelpers::ExpandToChildrenFromOffsets<float>("
    "td.trk.chi2TPC, trackInfo.idxFirstResidual, (int)res.dy.size())")
print("  Expanded: exp_dEdxTPC, exp_chi2TPC (track→cluster level)")

# --- Original parent columns as standalone (for snapshot) ---
rdf2 = rdf2.Define("trk_dEdxTPC", "td.trk.dEdxTPC")
rdf2 = rdf2.Define("trk_chi2TPC", "td.trk.chi2TPC")
rdf2 = rdf2.Define("offsets", "trackInfo.idxFirstResidual")
print("  Copied: trk_dEdxTPC, trk_chi2TPC, offsets (for snapshot)")

print(f"\n[OK] E2 PASSED")

# ============================================================
# E3: Snapshot — All Defined Columns (no raw O2 objects)
# ============================================================
print("\n" + "=" * 60)
print("E3: Snapshot to ROOT file")
print("=" * 60)

snap_cols = ROOT.std.vector["string"]()
snap_list = [
    # Child-level (all same length per event)
    "dy", "dz", "row", "sec", "qMaxTPC", "qTotTPC",
    # Parent index + expanded (same length as children)
    "parentIdx", "exp_dEdxTPC", "exp_chi2TPC",
    # Original parent-level (track length per event)
    "trk_dEdxTPC", "trk_chi2TPC", "offsets",
]
for c in snap_list:
    snap_cols.push_back(c)

rdf2.Snapshot("expanded", SNAPSHOT_FILE, snap_cols)
size_mb = os.path.getsize(SNAPSHOT_FILE) / 1e6
print(f"  Saved: {SNAPSHOT_FILE} ({size_mb:.1f} MB)")
print(f"  Tree 'expanded': {len(snap_list)} columns")
print(f"    Child-level:    dy, dz, row, sec, qMaxTPC, qTotTPC")
print(f"    Expanded:       exp_dEdxTPC, exp_chi2TPC, parentIdx")
print(f"    Parent-level:   trk_dEdxTPC, trk_chi2TPC, offsets")

print(f"\n[OK] E3 PASSED")

# ============================================================
# E4: Readback — Cross-Level Queries from Snapshot
# ============================================================
print("\n" + "=" * 60)
print("E4: Readback from snapshot")
print("=" * 60)

rdf_snap = ROOT.RDataFrame("expanded", SNAPSHOT_FILE)
n_entries = rdf_snap.Count().GetValue()
print(f"  Events: {n_entries}")

# --- A: The query that was impossible: qMaxTPC vs expanded dEdxTPC ---
print(f"\n  A) qMaxTPC : exp_dEdxTPC (the cross-level query)")
data_cross = rdf_snap.AsNumpy(["qMaxTPC", "exp_dEdxTPC"])
for i in range(min(3, n_entries)):
    q = np.array(data_cross["qMaxTPC"][i], dtype=np.float64)
    d = np.array(data_cross["exp_dEdxTPC"][i], dtype=np.float64)
    assert len(q) == len(d), f"Event {i}: qMaxTPC({len(q)}) != exp_dEdxTPC({len(d)})"
    corr = np.corrcoef(q, d)[0, 1] if len(q) > 1 else float('nan')
    print(f"     Event {i}: {len(q)} points, corr(qMax, dEdx) = {corr:.4f}")

# --- B: Parent-level still queryable ---
print(f"\n  B) Original parent-level columns:")
data_trk = rdf_snap.AsNumpy(["trk_dEdxTPC", "offsets"])
for i in range(min(3, n_entries)):
    nt = len(data_trk["trk_dEdxTPC"][i])
    no = len(data_trk["offsets"][i])
    print(f"     Event {i}: {nt} tracks, {no} offsets")

# --- C: dy vs expanded dEdx (residual vs track property) ---
print(f"\n  C) dy : exp_dEdxTPC (residual position vs track energy loss)")
data_mix = rdf_snap.AsNumpy(["dy", "exp_dEdxTPC"])
total = sum(len(data_mix["dy"][i]) for i in range(n_entries))
print(f"     Total points: {total}")

print(f"\n[OK] E4 PASSED")

# ============================================================
# E5: DSL Path — register_parent_child → to_pandas
# ============================================================
print("\n" + "=" * 60)
print("E5: DSL auto-expansion path")
print("=" * 60)

from RDataFrameDSL import DSLCompiler

# Fresh RDF
chain3 = ROOT.TChain("unbinnedResid")
chain3.Add(RESIDUALS_FILE)
chain_td3 = ROOT.TChain("trackData")
chain_td3.Add(RESIDUALS_FILE)
chain3.AddFriend(chain_td3, "td")
rdf3 = ROOT.RDataFrame(chain3).Range(N_EVENTS)

schema = {
    "res.dy":    "RVec<short>",
    "res.dz":    "RVec<short>",
    "res.row":   "RVec<unsigned char>",
    "td.trk.dEdxTPC":  "RVec<float>",
    "td.trk.chi2TPC":  "RVec<float>",
    "trackInfo.idxFirstResidual": "RVec<int>",
}

dsl = DSLCompiler(schema)
dsl.register_parent_child(
    parent="td.trk",
    child="res",
    offset_column="trackInfo.idxFirstResidual"
)

df_dsl = dsl.to_pandas(
    rdf3,
    columns=["res.dy", "res.dz", "td.trk.dEdxTPC", "td.trk.chi2TPC"],
    parent_id_column=None
)

print(f"  DSL result: {df_dsl.shape[0]} rows × {df_dsl.shape[1]} columns")
print(f"  Columns: {list(df_dsl.columns)}")
print(f"  corr(res.dy, td.trk.dEdxTPC) = "
      f"{df_dsl['res.dy'].astype(float).corr(df_dsl['td.trk.dEdxTPC']):.4f}")

print(f"\n[OK] E5 PASSED")

# ============================================================
# E6: Invariance Tests
# ============================================================
print("\n" + "=" * 60)
print("E6: Invariance Tests")
print("=" * 60)

# Load all snapshot data for validation
data_all = rdf_snap.AsNumpy([
    "dy", "dz", "qMaxTPC", "exp_dEdxTPC", "exp_chi2TPC",
    "parentIdx", "trk_dEdxTPC", "offsets",
])

n_pass = 0
n_test = 0

# --- INV1: len(child) == len(expanded) == len(parentIdx) ---
n_test += 1
for evt in range(n_entries):
    nc = len(data_all["dy"][evt])
    ne = len(data_all["exp_dEdxTPC"][evt])
    np_ = len(data_all["parentIdx"][evt])
    nq = len(data_all["qMaxTPC"][evt])
    assert nc == ne == np_ == nq, \
        f"Event {evt}: dy={nc}, exp={ne}, pidx={np_}, qMax={nq}"
total_rows = sum(len(data_all["dy"][i]) for i in range(n_entries))
print(f"  INV1: len(child)==len(expanded)==len(parentIdx)==len(qMaxTPC) "
      f"({total_rows}) ✓")
n_pass += 1

# --- INV2: expanded[i] == parent[parentIdx[i]] ---
n_test += 1
for evt in range(n_entries):
    pidx = np.array(data_all["parentIdx"][evt], dtype=np.int32)
    parent = np.array(data_all["trk_dEdxTPC"][evt], dtype=np.float32)
    expanded = np.array(data_all["exp_dEdxTPC"][evt], dtype=np.float32)
    expected = parent[pidx]
    np.testing.assert_allclose(expanded, expected, rtol=1e-5,
        err_msg=f"Event {evt}: expanded != parent[parentIdx]")
print(f"  INV2: expanded[i] == parent[parentIdx[i]] for all events ✓")
n_pass += 1

# --- INV3: sum(expanded) == sum(value × child_count) ---
n_test += 1
for evt in range(n_entries):
    offsets = np.array(data_all["offsets"][evt], dtype=np.int32)
    parent = np.array(data_all["trk_dEdxTPC"][evt], dtype=np.float64)
    expanded = np.array(data_all["exp_dEdxTPC"][evt], dtype=np.float64)
    n_ch = len(data_all["dy"][evt])
    child_counts = np.diff(offsets, append=n_ch)
    weighted = np.sum(parent * child_counts)
    exp_sum = np.sum(expanded)
    np.testing.assert_allclose(exp_sum, weighted, rtol=1e-4,
        err_msg=f"Event {evt}: sum mismatch")
print(f"  INV3: Σ(expanded) == Σ(value × child_count) ✓")
n_pass += 1

# --- INV4: expanded constant within each track ---
n_test += 1
for evt in range(n_entries):
    pidx = np.array(data_all["parentIdx"][evt], dtype=np.int32)
    expanded = np.array(data_all["exp_dEdxTPC"][evt], dtype=np.float32)
    for tid in np.unique(pidx):
        vals = expanded[pidx == tid]
        assert np.all(vals == vals[0]), \
            f"Event {evt}, track {tid}: not constant"
print(f"  INV4: expanded constant within each track ✓")
n_pass += 1

# --- INV5: DSL to_pandas == Snapshot expanded ---
n_test += 1
snap_dEdx = np.concatenate([
    np.array(data_all["exp_dEdxTPC"][i], dtype=np.float32)
    for i in range(n_entries)
])
dsl_dEdx = df_dsl["td.trk.dEdxTPC"].values.astype(np.float32)
assert len(dsl_dEdx) == len(snap_dEdx), \
    f"Row count: DSL={len(dsl_dEdx)}, snap={len(snap_dEdx)}"
np.testing.assert_allclose(dsl_dEdx, snap_dEdx, rtol=1e-5)
print(f"  INV5: DSL to_pandas == Snapshot expanded ({len(dsl_dEdx)} values) ✓")
n_pass += 1

# --- INV6: qMaxTPC length == expanded length ---
n_test += 1
for evt in range(n_entries):
    nq = len(data_all["qMaxTPC"][evt])
    ne = len(data_all["exp_dEdxTPC"][evt])
    assert nq == ne, f"Event {evt}: qMaxTPC({nq}) != exp({ne})"
print(f"  INV6: len(qMaxTPC) == len(exp_dEdxTPC) for all events ✓")
n_pass += 1

# --- INV7: parentIdx monotonically non-decreasing per event ---
n_test += 1
for evt in range(n_entries):
    pidx = np.array(data_all["parentIdx"][evt], dtype=np.int32)
    assert np.all(np.diff(pidx) >= 0), \
        f"Event {evt}: parentIdx not monotonic"
print(f"  INV7: parentIdx monotonically non-decreasing ✓")
n_pass += 1

# --- INV8: max(parentIdx) + 1 == number of tracks ---
n_test += 1
for evt in range(n_entries):
    pidx = np.array(data_all["parentIdx"][evt], dtype=np.int32)
    n_tracks = len(data_all["trk_dEdxTPC"][evt])
    assert pidx.max() + 1 == n_tracks, \
        f"Event {evt}: max(pidx)={pidx.max()}, n_tracks={n_tracks}"
print(f"  INV8: max(parentIdx)+1 == n_tracks ✓")
n_pass += 1

print(f"\n  {n_pass}/{n_test} invariance tests PASSED")
print(f"\n[OK] E6 PASSED")

# ============================================================
# E7: Summary
# ============================================================
print("\n" + "=" * 60)
print("E7: Summary")
print("=" * 60)

print(f"""
The Query TTree::Draw Cannot Do:
  qMaxTPC : td.trk.dEdxTPC
  (cluster charge vs track energy loss — different array lengths)

Data:
  {N_EVENTS} events, {total_children} clusters, {total_parents} tracks
  ~{total_children//total_parents} clusters per track

Snapshot ({SNAPSHOT_FILE}, {size_mb:.1f} MB):
  Child-level:   dy, dz, row, sec, qMaxTPC, qTotTPC
  Expanded:      exp_dEdxTPC, exp_chi2TPC, parentIdx
  Parent-level:  trk_dEdxTPC, trk_chi2TPC, offsets
  → Query both levels from same file

DSL path:
  dsl.register_parent_child("td.trk", "res", "trackInfo.idxFirstResidual")
  df = dsl.to_pandas(rdf, ["res.dy", "td.trk.dEdxTPC"])
  → {df_dsl.shape[0]} rows, automatic expansion

Invariance: {n_pass}/{n_test} tests passed
""")
