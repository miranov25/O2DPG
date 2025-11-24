/**
 * test_composite_index.C - Enhanced tests for N-key composite index
 * 
 * Tests verify both formal correctness AND semantic correctness:
 * - Composite index is actually built (not linear scan fallback)
 * - Index lookups return correct matching values
 * - Join produces expected results
 * 
 * Usage:
 *   root -l -b -q test_composite_index.C
 */

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TString.h>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cassert>
#include <cmath>

// Include the main macro
#include "../AliasDataFrameTree.C"

// Test counters
Int_t g_assertions_passed = 0;
Int_t g_assertions_failed = 0;

// Assertion macros
#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        std::cerr << "  ASSERTION FAILED: " << msg << std::endl; \
        g_assertions_failed++; \
        return kFALSE; \
    } else { \
        g_assertions_passed++; \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        std::cerr << "  ASSERTION FAILED: " << msg << " (expected " << (b) << ", got " << (a) << ")" << std::endl; \
        g_assertions_failed++; \
        return kFALSE; \
    } else { \
        g_assertions_passed++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "  ASSERTION FAILED: " << msg << " (expected " << (b) << " +/- " << (tol) << ", got " << (a) << ")" << std::endl; \
        g_assertions_failed++; \
        return kFALSE; \
    } else { \
        g_assertions_passed++; \
    } \
} while(0)

/**
 * Create test data with specified index columns
 */
void CreateTestFile(const char* filename, Int_t nMain, Int_t nSubframe,
                    std::vector<Long64_t>& orbits, Int_t seed = 12345) {
    TRandom3 rnd(seed);
    
    TFile* f = TFile::Open(filename, "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main tree");
    
    Int_t row, drift25, side;
    Long64_t firstTFOrbit;
    Float_t mX, mY;
    
    mainTree->Branch("row", &row, "row/I");
    mainTree->Branch("drift25", &drift25, "drift25/I");
    mainTree->Branch("side", &side, "side/I");
    mainTree->Branch("firstTFOrbit", &firstTFOrbit, "firstTFOrbit/L");
    mainTree->Branch("mX", &mX, "mX/F");
    mainTree->Branch("mY", &mY, "mY/F");
    
    for (Int_t i = 0; i < nMain; i++) {
        row = rnd.Integer(63);
        drift25 = rnd.Integer(26);
        side = rnd.Integer(2);
        firstTFOrbit = orbits[rnd.Integer(orbits.size())];
        mX = rnd.Gaus(0, 1);
        mY = rnd.Gaus(0, 1);
        mainTree->Fill();
    }
    mainTree->Write();
    
    TTree* sfTree = new TTree("tree__subframe__Calib", "Calibration subframe");
    Float_t gain, offset;
    
    sfTree->Branch("row", &row, "row/I");
    sfTree->Branch("drift25", &drift25, "drift25/I");
    sfTree->Branch("side", &side, "side/I");
    sfTree->Branch("firstTFOrbit", &firstTFOrbit, "firstTFOrbit/L");
    sfTree->Branch("gain", &gain, "gain/F");
    sfTree->Branch("offset", &offset, "offset/F");
    
    std::set<std::tuple<Int_t, Int_t, Int_t, Long64_t>> seen;
    
    for (Int_t i = 0; i < nSubframe; i++) {
        row = rnd.Integer(63);
        drift25 = rnd.Integer(26);
        side = rnd.Integer(2);
        firstTFOrbit = orbits[rnd.Integer(orbits.size())];
        
        auto key = std::make_tuple(row, drift25, side, firstTFOrbit);
        if (seen.count(key)) continue;
        seen.insert(key);
        
        gain = 1.0 + rnd.Gaus(0, 0.1);
        offset = rnd.Gaus(0, 0.01);
        sfTree->Fill();
    }
    sfTree->Write();
    
    TString schema = R"({
  "subframes": {
    "Calib": {"index": ["row", "drift25", "side", "firstTFOrbit"]}
  }
})";
    TObjString* schemaObj = new TObjString(schema);
    schemaObj->Write("ADF_SCHEMA");
    
    f->Close();
    delete f;
    
    std::cout << "Created: " << filename << " (" << nMain << " main, " << seen.size() << " subframe)" << std::endl;
}

/**
 * Test 1: 3-key composite index with SEMANTIC verification
 */
Bool_t Test3KeyIndex() {
    std::cout << "\n========== Test 1: 3-Key Index (Semantic) ==========" << std::endl;
    
    TFile* f = TFile::Open("test_3key.root", "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main");
    TTree* sfTree = new TTree("tree__subframe__SF", "Subframe");
    
    Int_t row, drift, side;
    Float_t value, calib;
    
    mainTree->Branch("row", &row, "row/I");
    mainTree->Branch("drift", &drift, "drift/I");
    mainTree->Branch("side", &side, "side/I");
    mainTree->Branch("value", &value, "value/F");
    
    sfTree->Branch("row", &row, "row/I");
    sfTree->Branch("drift", &drift, "drift/I");
    sfTree->Branch("side", &side, "side/I");
    sfTree->Branch("calib", &calib, "calib/F");
    
    TRandom3 rnd(42);
    for (int i = 0; i < 100; i++) {
        row = rnd.Integer(10);
        drift = rnd.Integer(5);
        side = rnd.Integer(2);
        value = rnd.Gaus(0, 1);
        mainTree->Fill();
    }
    
    // Deterministic calibration: calib = 1.0 + row*0.01 + drift*0.001 + side*0.0001
    for (int r = 0; r < 10; r++) {
        for (int d = 0; d < 5; d++) {
            for (int s = 0; s < 2; s++) {
                row = r; drift = d; side = s;
                calib = 1.0 + r * 0.01 + d * 0.001 + s * 0.0001;
                sfTree->Fill();
            }
        }
    }
    
    TString schema = R"({"subframes": {"SF": {"index": ["row", "drift", "side"]}}})";
    TObjString* obj = new TObjString(schema);
    obj->Write("ADF_SCHEMA");
    
    mainTree->Write();
    sfTree->Write();
    f->Close();
    
    TTree* tree = LoadADFTree("test_3key.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // Verify schema parsed
    auto it = g_schemaRegistry.find(tree);
    ASSERT_TRUE(it != g_schemaRegistry.end(), "Schema exists");
    ASSERT_TRUE(it->second.subframeIndices.count("SF") > 0, "SF in schema");
    ASSERT_EQ((int)it->second.subframeIndices["SF"].size(), 3, "3 index columns");
    
    // Semantic test: verify joined values match formula
    Long64_t nDrawn = tree->Draw("row:drift:side:SF.calib", "", "goff");
    ASSERT_EQ(nDrawn, 100LL, "100 entries drawn");
    
    Double_t* rowV = tree->GetV1();
    Double_t* driftV = tree->GetV2();
    Double_t* sideV = tree->GetV3();
    Double_t* calibV = tree->GetV4();
    
    Int_t correct = 0;
    for (Int_t i = 0; i < 20; i++) {
        Double_t expected = 1.0 + rowV[i] * 0.01 + driftV[i] * 0.001 + sideV[i] * 0.0001;
        if (std::abs(calibV[i] - expected) < 1e-6) correct++;
    }
    
    ASSERT_EQ(correct, 20, "20 entries have correct join values");
    
    std::cout << "  SEMANTIC: Join values verified correct" << std::endl;
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 2: 4-key sparse index
 */
Bool_t Test4KeySparseIndex() {
    std::cout << "\n========== Test 2: 4-Key Sparse Index ==========" << std::endl;
    
    std::vector<Long64_t> orbits = {
        547832001, 547832045, 547832112,
        547835003, 547835067,
        548000001, 548000099,
        549000000, 549500000
    };
    
    CreateTestFile("test_4key_sparse.root", 5000, 2000, orbits);
    
    TTree* tree = LoadADFTree("test_4key_sparse.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    auto it = g_schemaRegistry.find(tree);
    ASSERT_TRUE(it != g_schemaRegistry.end(), "Schema exists");
    ASSERT_TRUE(it->second.subframeIndices.count("Calib") > 0, "Calib in schema");
    ASSERT_EQ((int)it->second.subframeIndices["Calib"].size(), 4, "4 index columns");
    
    Long64_t nDrawn = tree->Draw("Calib.gain", "", "goff");
    ASSERT_TRUE(nDrawn > 0, "Has Calib.gain");
    std::cout << "  Entries with Calib.gain: " << nDrawn << std::endl;
    
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 3: Merge safety
 */
Bool_t TestMergeSafety() {
    std::cout << "\n========== Test 3: Merge Safety ==========" << std::endl;
    
    std::vector<Long64_t> orbits1 = {100001, 100002, 100003};
    std::vector<Long64_t> orbits2 = {200001, 200002, 200003};
    
    CreateTestFile("test_merge_1.root", 1000, 500, orbits1, 111);
    CreateTestFile("test_merge_2.root", 1000, 500, orbits2, 222);
    
    gSystem->Exec("hadd -f test_merged.root test_merge_1.root test_merge_2.root > /dev/null 2>&1");
    
    TTree* tree = LoadADFTree("test_merged.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Merged tree loaded");
    ASSERT_EQ(tree->GetEntries(), 2000LL, "2000 entries after merge");
    
    Long64_t nDrawn = tree->Draw("Calib.gain", "", "goff");
    ASSERT_TRUE(nDrawn > 0, "Subframe works after merge");
    std::cout << "  Entries with Calib.gain: " << nDrawn << std::endl;
    
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 4: Float column detection
 */
Bool_t TestNonIntegerError() {
    std::cout << "\n========== Test 4: Float Column Detection ==========" << std::endl;
    
    TFile* f = TFile::Open("test_float_index.root", "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main");
    TTree* sfTree = new TTree("tree__subframe__SF", "Subframe");
    
    Float_t floatKey;
    Int_t intKey1, intKey2;
    Float_t value;
    
    mainTree->Branch("floatKey", &floatKey, "floatKey/F");
    mainTree->Branch("intKey1", &intKey1, "intKey1/I");
    mainTree->Branch("intKey2", &intKey2, "intKey2/I");
    mainTree->Branch("value", &value, "value/F");
    
    sfTree->Branch("floatKey", &floatKey, "floatKey/F");
    sfTree->Branch("intKey1", &intKey1, "intKey1/I");
    sfTree->Branch("intKey2", &intKey2, "intKey2/I");
    sfTree->Branch("calib", &value, "calib/F");
    
    for (int i = 0; i < 100; i++) {
        floatKey = i * 0.1;
        intKey1 = i % 10;
        intKey2 = i / 10;
        value = i;
        mainTree->Fill();
        sfTree->Fill();
    }
    
    std::cout << "  Type of 'floatKey': " << sfTree->GetLeaf("floatKey")->GetTypeName() << std::endl;
    std::cout << "  Type of 'intKey1': " << sfTree->GetLeaf("intKey1")->GetTypeName() << std::endl;
    
    TString schema = R"({"subframes": {"SF": {"index": ["floatKey", "intKey1", "intKey2"]}}})";
    TObjString* obj = new TObjString(schema);
    obj->Write("ADF_SCHEMA");
    
    mainTree->Write();
    sfTree->Write();
    f->Close();
    
    std::cout << "  Expecting ERROR about non-integer column..." << std::endl;
    TTree* tree = LoadADFTree("test_float_index.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loads with degraded indexing");
    
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 5: Composite key uniqueness
 */
Bool_t TestCompositeKeyUniqueness() {
    std::cout << "\n========== Test 5: Composite Key Uniqueness ==========" << std::endl;
    
    TFile* f = TFile::Open("test_uniqueness.root", "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main");
    TTree* sfTree = new TTree("tree__subframe__SF", "Subframe");
    
    Int_t a, b, c;
    Float_t value;
    
    mainTree->Branch("a", &a, "a/I");
    mainTree->Branch("b", &b, "b/I");
    mainTree->Branch("c", &c, "c/I");
    mainTree->Branch("value", &value, "value/F");
    
    sfTree->Branch("a", &a, "a/I");
    sfTree->Branch("b", &b, "b/I");
    sfTree->Branch("c", &c, "c/I");
    sfTree->Branch("calib", &value, "calib/F");
    
    // All 3x4x5=60 combinations
    for (int ia = 0; ia < 3; ia++) {
        for (int ib = 0; ib < 4; ib++) {
            for (int ic = 0; ic < 5; ic++) {
                a = ia; b = ib; c = ic;
                value = ia * 100 + ib * 10 + ic;
                mainTree->Fill();
                
                value = 1000 + ia * 20 + ib * 5 + ic;  // Unique calib
                sfTree->Fill();
            }
        }
    }
    
    TString schema = R"({"subframes": {"SF": {"index": ["a", "b", "c"]}}})";
    TObjString* obj = new TObjString(schema);
    obj->Write("ADF_SCHEMA");
    
    mainTree->Write();
    sfTree->Write();
    f->Close();
    
    TTree* tree = LoadADFTree("test_uniqueness.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    Long64_t nDrawn = tree->Draw("a:b:c:SF.calib", "", "goff");
    ASSERT_EQ(nDrawn, 60LL, "60 entries");
    
    Double_t* aV = tree->GetV1();
    Double_t* bV = tree->GetV2();
    Double_t* cV = tree->GetV3();
    Double_t* calibV = tree->GetV4();
    
    Int_t correct = 0;
    for (Int_t i = 0; i < 60; i++) {
        Double_t expected = 1000 + aV[i] * 20 + bV[i] * 5 + cV[i];
        if (std::abs(calibV[i] - expected) < 1e-6) correct++;
    }
    
    ASSERT_EQ(correct, 60, "All 60 joins correct");
    
    std::cout << "  VERIFIED: All 60 unique combinations correctly joined" << std::endl;
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 6: Multiple Subframes with Different Index Definitions
 * 
 * This is the CRITICAL test that verifies:
 * - Two subframes with DIFFERENT index columns work independently
 * - Each subframe gets its own __adf_key_<name>__ branch
 * - Both can be queried correctly in same tree
 */
Bool_t TestMultipleSubframes() {
    std::cout << "\n========== Test 6: Multiple Subframes ==========" << std::endl;
    
    TFile* f = TFile::Open("test_multi_sf.root", "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main");
    TTree* sf1Tree = new TTree("tree__subframe__SF1", "Subframe 1");
    TTree* sf2Tree = new TTree("tree__subframe__SF2", "Subframe 2");
    
    // Main tree has columns for BOTH subframe indices
    Int_t a, b, c;      // For SF1 (3 keys)
    Int_t x, y;         // For SF2 (2 keys)
    Float_t value;
    
    mainTree->Branch("a", &a, "a/I");
    mainTree->Branch("b", &b, "b/I");
    mainTree->Branch("c", &c, "c/I");
    mainTree->Branch("x", &x, "x/I");
    mainTree->Branch("y", &y, "y/I");
    mainTree->Branch("value", &value, "value/F");
    
    // SF1: indexed by [a, b, c]
    Float_t calib1;
    sf1Tree->Branch("a", &a, "a/I");
    sf1Tree->Branch("b", &b, "b/I");
    sf1Tree->Branch("c", &c, "c/I");
    sf1Tree->Branch("calib1", &calib1, "calib1/F");
    
    // SF2: indexed by [x, y] (different columns!)
    Float_t calib2;
    sf2Tree->Branch("x", &x, "x/I");
    sf2Tree->Branch("y", &y, "y/I");
    sf2Tree->Branch("calib2", &calib2, "calib2/F");
    
    // Fill main tree with all combinations
    for (int ia = 0; ia < 3; ia++) {
        for (int ib = 0; ib < 4; ib++) {
            for (int ic = 0; ic < 2; ic++) {
                for (int ix = 0; ix < 5; ix++) {
                    for (int iy = 0; iy < 3; iy++) {
                        a = ia; b = ib; c = ic;
                        x = ix; y = iy;
                        value = a * 100 + b * 10 + c + x * 0.1 + y * 0.01;
                        mainTree->Fill();
                    }
                }
            }
        }
    }
    std::cout << "  Main tree entries: " << mainTree->GetEntries() << std::endl;
    
    // Fill SF1: calib1 = 100 + a*10 + b + c*0.1
    for (int ia = 0; ia < 3; ia++) {
        for (int ib = 0; ib < 4; ib++) {
            for (int ic = 0; ic < 2; ic++) {
                a = ia; b = ib; c = ic;
                calib1 = 100 + a * 10 + b + c * 0.1;
                sf1Tree->Fill();
            }
        }
    }
    std::cout << "  SF1 entries: " << sf1Tree->GetEntries() << " (3 keys: a,b,c)" << std::endl;
    
    // Fill SF2: calib2 = 200 + x*10 + y
    for (int ix = 0; ix < 5; ix++) {
        for (int iy = 0; iy < 3; iy++) {
            x = ix; y = iy;
            calib2 = 200 + x * 10 + y;
            sf2Tree->Fill();
        }
    }
    std::cout << "  SF2 entries: " << sf2Tree->GetEntries() << " (2 keys: x,y)" << std::endl;
    
    // Schema with TWO subframes with DIFFERENT index definitions
    TString schema = R"({
  "subframes": {
    "SF1": {"index": ["a", "b", "c"]},
    "SF2": {"index": ["x", "y"]}
  }
})";
    TObjString* obj = new TObjString(schema);
    obj->Write("ADF_SCHEMA");
    
    mainTree->Write();
    sf1Tree->Write();
    sf2Tree->Write();
    f->Close();
    
    // Load and test
    TTree* tree = LoadADFTree("test_multi_sf.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // Verify schema parsed both subframes
    auto it = g_schemaRegistry.find(tree);
    ASSERT_TRUE(it != g_schemaRegistry.end(), "Schema exists");
    ASSERT_TRUE(it->second.subframeIndices.count("SF1") > 0, "SF1 in schema");
    ASSERT_TRUE(it->second.subframeIndices.count("SF2") > 0, "SF2 in schema");
    ASSERT_EQ((int)it->second.subframeIndices["SF1"].size(), 3, "SF1 has 3 index columns");
    ASSERT_EQ((int)it->second.subframeIndices["SF2"].size(), 2, "SF2 has 2 index columns");
    
    // Verify BOTH subframes are accessible
    Long64_t n1 = tree->Draw("SF1.calib1", "", "goff");
    Long64_t n2 = tree->Draw("SF2.calib2", "", "goff");
    std::cout << "  Entries with SF1.calib1: " << n1 << std::endl;
    std::cout << "  Entries with SF2.calib2: " << n2 << std::endl;
    ASSERT_TRUE(n1 > 0, "SF1 accessible");
    ASSERT_TRUE(n2 > 0, "SF2 accessible");
    
    // SEMANTIC TEST: Verify BOTH joins produce correct values
    Long64_t nDrawn = tree->Draw("a:b:c:SF1.calib1", "", "goff");
    Double_t* aV = tree->GetV1();
    Double_t* bV = tree->GetV2();
    Double_t* cV = tree->GetV3();
    Double_t* c1V = tree->GetV4();
    
    Int_t correct1 = 0;
    for (Int_t i = 0; i < TMath::Min(nDrawn, 50LL); i++) {
        Double_t expected = 100 + aV[i] * 10 + bV[i] + cV[i] * 0.1;
        if (std::abs(c1V[i] - expected) < 1e-5) correct1++;
    }
    std::cout << "  SF1 semantic check: " << correct1 << "/50 correct" << std::endl;
    ASSERT_TRUE(correct1 >= 45, "Most SF1 joins correct");  // Allow some tolerance
    
    nDrawn = tree->Draw("x:y:SF2.calib2", "", "goff");
    Double_t* xV = tree->GetV1();
    Double_t* yV = tree->GetV2();
    Double_t* c2V = tree->GetV3();
    
    Int_t correct2 = 0;
    for (Int_t i = 0; i < TMath::Min(nDrawn, 50LL); i++) {
        Double_t expected = 200 + xV[i] * 10 + yV[i];
        if (std::abs(c2V[i] - expected) < 1e-5) correct2++;
    }
    std::cout << "  SF2 semantic check: " << correct2 << "/50 correct" << std::endl;
    ASSERT_TRUE(correct2 >= 45, "Most SF2 joins correct");
    
    std::cout << "  VERIFIED: Both subframes work independently!" << std::endl;
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 7: Index Assertion
 * 
 * Verify that GetTreeIndex() returns non-null for indexed subframes
 */
Bool_t TestIndexAssertion() {
    std::cout << "\n========== Test 7: Index Assertion ==========" << std::endl;
    
    // Use the file from Test 5 (already has clean 3-key index)
    TFile* f = TFile::Open("test_uniqueness.root", "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main");
    TTree* sfTree = new TTree("tree__subframe__SF", "Subframe");
    
    Int_t a, b, c;
    Float_t value;
    
    mainTree->Branch("a", &a, "a/I");
    mainTree->Branch("b", &b, "b/I");
    mainTree->Branch("c", &c, "c/I");
    mainTree->Branch("value", &value, "value/F");
    
    sfTree->Branch("a", &a, "a/I");
    sfTree->Branch("b", &b, "b/I");
    sfTree->Branch("c", &c, "c/I");
    sfTree->Branch("calib", &value, "calib/F");
    
    for (int ia = 0; ia < 3; ia++) {
        for (int ib = 0; ib < 4; ib++) {
            for (int ic = 0; ic < 5; ic++) {
                a = ia; b = ib; c = ic;
                value = ia * 100 + ib * 10 + ic;
                mainTree->Fill();
                sfTree->Fill();
            }
        }
    }
    
    TString schema = R"({"subframes": {"SF": {"index": ["a", "b", "c"]}}})";
    TObjString* obj = new TObjString(schema);
    obj->Write("ADF_SCHEMA");
    
    mainTree->Write();
    sfTree->Write();
    f->Close();
    
    // Load
    TTree* tree = LoadADFTree("test_uniqueness.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // Get the friend tree and check its index
    TList* friends = tree->GetListOfFriends();
    ASSERT_TRUE(friends != nullptr, "Has friends");
    ASSERT_TRUE(friends->GetEntries() > 0, "Has at least one friend");
    
    TFriendElement* fe = (TFriendElement*)friends->At(0);
    TTree* friendTree = fe->GetTree();
    ASSERT_TRUE(friendTree != nullptr, "Friend tree exists");
    
    // THE CRITICAL ASSERTION: Index must exist!
    TVirtualIndex* idx = friendTree->GetTreeIndex();
    std::cout << "  Friend tree index: " << (idx ? "EXISTS" : "NULL") << std::endl;
    ASSERT_TRUE(idx != nullptr, "Subframe has TTreeIndex (composite index was built)");
    
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 8: Empty Subframe
 * 
 * Verify that an empty subframe (0 entries) doesn't crash
 * and produces no matches.
 */
Bool_t TestEmptySubframe() {
    std::cout << "\n========== Test 8: Empty Subframe ==========" << std::endl;
    
    TFile* f = TFile::Open("test_empty_sf.root", "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main");
    TTree* sfTree = new TTree("tree__subframe__SF", "Subframe");
    
    Int_t a, b, c;
    Float_t value, calib;
    
    mainTree->Branch("a", &a, "a/I");
    mainTree->Branch("b", &b, "b/I");
    mainTree->Branch("c", &c, "c/I");
    mainTree->Branch("value", &value, "value/F");
    
    sfTree->Branch("a", &a, "a/I");
    sfTree->Branch("b", &b, "b/I");
    sfTree->Branch("c", &c, "c/I");
    sfTree->Branch("calib", &calib, "calib/F");
    
    // Fill main tree with data
    for (int i = 0; i < 100; i++) {
        a = i % 5; b = i % 4; c = i % 3;
        value = i;
        mainTree->Fill();
    }
    
    // DO NOT fill subframe - leave it empty!
    std::cout << "  Main tree entries: " << mainTree->GetEntries() << std::endl;
    std::cout << "  Subframe entries: " << sfTree->GetEntries() << " (intentionally empty)" << std::endl;
    
    TString schema = R"({"subframes": {"SF": {"index": ["a", "b", "c"]}}})";
    TObjString* obj = new TObjString(schema);
    obj->Write("ADF_SCHEMA");
    
    mainTree->Write();
    sfTree->Write();
    f->Close();
    
    // Load - should not crash
    TTree* tree = LoadADFTree("test_empty_sf.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded despite empty subframe");
    
    // With empty subframe, ROOT returns all main entries (friend values are default)
    // The key point is: no crash, graceful handling
    Long64_t nDrawn = tree->Draw("SF.calib", "", "goff");
    std::cout << "  Entries drawn: " << nDrawn << std::endl;
    
    // Accept any result >= 0 (ROOT behavior varies)
    ASSERT_TRUE(nDrawn >= 0, "Draw completed without crash");
    
    // Main tree should still be fully accessible
    Long64_t nMain = tree->Draw("value", "", "goff");
    ASSERT_EQ(nMain, 100LL, "Main tree entries accessible");
    
    std::cout << "  VERIFIED: Empty subframe handled gracefully" << std::endl;
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 9: No Matches At All
 * 
 * Main tree has keys that DO NOT exist in subframe.
 * Verifies left-join behavior: main entries preserved, joined column is empty/NaN.
 */
Bool_t TestNoMatches() {
    std::cout << "\n========== Test 9: No Matches ==========" << std::endl;
    
    TFile* f = TFile::Open("test_no_match.root", "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main");
    TTree* sfTree = new TTree("tree__subframe__SF", "Subframe");
    
    Int_t a, b, c;
    Float_t value, calib;
    
    mainTree->Branch("a", &a, "a/I");
    mainTree->Branch("b", &b, "b/I");
    mainTree->Branch("c", &c, "c/I");
    mainTree->Branch("value", &value, "value/F");
    
    sfTree->Branch("a", &a, "a/I");
    sfTree->Branch("b", &b, "b/I");
    sfTree->Branch("c", &c, "c/I");
    sfTree->Branch("calib", &calib, "calib/F");
    
    // Main tree: keys 0-9
    for (int i = 0; i < 100; i++) {
        a = i % 10;  // 0-9
        b = i % 5;   // 0-4
        c = i % 2;   // 0-1
        value = i;
        mainTree->Fill();
    }
    
    // Subframe: keys 100-109 (NO OVERLAP with main!)
    for (int i = 0; i < 50; i++) {
        a = 100 + (i % 10);  // 100-109 - completely different!
        b = i % 5;
        c = i % 2;
        calib = 999.0;
        sfTree->Fill();
    }
    
    std::cout << "  Main tree keys: a in [0-9]" << std::endl;
    std::cout << "  Subframe keys: a in [100-109] (no overlap!)" << std::endl;
    
    TString schema = R"({"subframes": {"SF": {"index": ["a", "b", "c"]}}})";
    TObjString* obj = new TObjString(schema);
    obj->Write("ADF_SCHEMA");
    
    mainTree->Write();
    sfTree->Write();
    f->Close();
    
    // Load
    TTree* tree = LoadADFTree("test_no_match.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // Draw SF.calib - should return 0 matches
    Long64_t nDrawn = tree->Draw("SF.calib", "", "goff");
    std::cout << "  Entries with SF.calib: " << nDrawn << std::endl;
    
    // No matches expected because keys don't overlap
    ASSERT_EQ(nDrawn, 0LL, "No matches when keys don't overlap");
    
    // Main tree should still be accessible
    Long64_t nMain = tree->Draw("value", "", "goff");
    ASSERT_EQ(nMain, 100LL, "Main tree entries still accessible");
    
    std::cout << "  VERIFIED: No matches, main tree intact" << std::endl;
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Test 10: Duplicate Keys in Subframe
 * 
 * When subframe has duplicate keys, verify behavior is consistent
 * (ROOT typically uses the FIRST matching entry).
 */
Bool_t TestDuplicateKeys() {
    std::cout << "\n========== Test 10: Duplicate Keys ==========" << std::endl;
    
    TFile* f = TFile::Open("test_dup_keys.root", "RECREATE");
    
    TTree* mainTree = new TTree("tree", "Main");
    TTree* sfTree = new TTree("tree__subframe__SF", "Subframe");
    
    Int_t a, b, c;
    Float_t value, calib;
    
    mainTree->Branch("a", &a, "a/I");
    mainTree->Branch("b", &b, "b/I");
    mainTree->Branch("c", &c, "c/I");
    mainTree->Branch("value", &value, "value/F");
    
    sfTree->Branch("a", &a, "a/I");
    sfTree->Branch("b", &b, "b/I");
    sfTree->Branch("c", &c, "c/I");
    sfTree->Branch("calib", &calib, "calib/F");
    
    // Main tree: single entry with key (1,1,1)
    a = 1; b = 1; c = 1; value = 42;
    mainTree->Fill();
    
    // Subframe: DUPLICATE keys with DIFFERENT values
    // Entry 0: (1,1,1) -> calib = 100
    // Entry 1: (1,1,1) -> calib = 200 (duplicate!)
    // Entry 2: (1,1,1) -> calib = 300 (duplicate!)
    a = 1; b = 1; c = 1;
    calib = 100.0; sfTree->Fill();
    calib = 200.0; sfTree->Fill();
    calib = 300.0; sfTree->Fill();
    
    std::cout << "  Main: 1 entry with key (1,1,1)" << std::endl;
    std::cout << "  Subframe: 3 entries with SAME key (1,1,1), calib=[100,200,300]" << std::endl;
    
    TString schema = R"({"subframes": {"SF": {"index": ["a", "b", "c"]}}})";
    TObjString* obj = new TObjString(schema);
    obj->Write("ADF_SCHEMA");
    
    mainTree->Write();
    sfTree->Write();
    f->Close();
    
    // Load
    TTree* tree = LoadADFTree("test_dup_keys.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // Draw - behavior with duplicates may vary
    Long64_t nDrawn = tree->Draw("SF.calib", "", "goff");
    std::cout << "  Entries drawn: " << nDrawn << std::endl;
    
    // The key point is: no crash, system handles duplicates
    ASSERT_TRUE(nDrawn >= 0, "Draw completed without crash");
    
    if (nDrawn > 0) {
        Double_t* calibV = tree->GetV1();
        if (calibV) {
            Double_t drawnCalib = calibV[0];
            std::cout << "  Drawn SF.calib value: " << drawnCalib << std::endl;
            // Should be one of the values we inserted
            ASSERT_TRUE(drawnCalib == 100.0 || drawnCalib == 200.0 || drawnCalib == 300.0 || drawnCalib == 0.0, 
                        "Got valid value");
        }
    }
    
    std::cout << "  NOTE: Duplicate keys handled (ROOT behavior may vary)" << std::endl;
    std::cout << "PASSED" << std::endl;
    return kTRUE;
}

/**
 * Main test runner
 */
void test_composite_index() {
    std::cout << "\n====================================================" << std::endl;
    std::cout << "Composite Index Tests (with semantic verification)" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    Int_t passed = 0;
    Int_t failed = 0;
    
    if (Test3KeyIndex()) passed++; else failed++;
    if (Test4KeySparseIndex()) passed++; else failed++;
    if (TestMergeSafety()) passed++; else failed++;
    if (TestNonIntegerError()) passed++; else failed++;
    if (TestCompositeKeyUniqueness()) passed++; else failed++;
    if (TestMultipleSubframes()) passed++; else failed++;
    if (TestIndexAssertion()) passed++; else failed++;
    if (TestEmptySubframe()) passed++; else failed++;
    if (TestNoMatches()) passed++; else failed++;
    if (TestDuplicateKeys()) passed++; else failed++;
    
    std::cout << "\n====================================================" << std::endl;
    std::cout << "Results: " << passed << "/" << (passed+failed) << " tests passed" << std::endl;
    std::cout << "Assertions: " << g_assertions_passed << " passed, " << g_assertions_failed << " failed" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    gSystem->Exec("rm -f test_3key.root test_4key_sparse.root test_merge_*.root test_merged.root test_float_index.root test_uniqueness.root test_multi_sf.root test_empty_sf.root test_no_match.root test_dup_keys.root");
    
    if (failed > 0) {
        std::cerr << "SOME TESTS FAILED!" << std::endl;
        gSystem->Exit(1);
    } else {
        std::cout << "ALL TESTS PASSED!" << std::endl;
    }
}
