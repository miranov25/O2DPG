/**
 * test_AliasDataFrameTree.C - Unit tests for AliasDataFrameTree.C
 * 
 * Run with: root -l -b -q test_AliasDataFrameTree.C
 * Or:       root -x test_AliasDataFrameTree.C
 * 
 * Prerequisites:
 * - AliasDataFrameTree.C in same directory or ROOT include path
 * - Test data file (will be created if not present)
 * 
 * Exit codes:
 *   0 = All tests passed
 *   1 = Test failure
 */

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TH1F.h>
#include <TString.h>
#include <iostream>
#include <cmath>
#include <vector>

// Include the macro we're testing
#include "../AliasDataFrameTree.C"

// Test configuration
const char* TEST_FILE = "/tmp/test_adf_tree.root";
const int N_MAIN = 1000;
const int N_SUB = 100;

// Test counters
int g_tests_passed = 0;
int g_tests_failed = 0;

// Helper macros
#define ASSERT_TRUE(cond, msg) \
    if (!(cond)) { \
        std::cerr << "FAIL: " << msg << " (line " << __LINE__ << ")" << std::endl; \
        g_tests_failed++; \
        return false; \
    } else { \
        g_tests_passed++; \
    }

#define ASSERT_EQ(a, b, msg) \
    if ((a) != (b)) { \
        std::cerr << "FAIL: " << msg << " (expected " << (b) << ", got " << (a) << ")" << std::endl; \
        g_tests_failed++; \
        return false; \
    } else { \
        g_tests_passed++; \
    }

#define ASSERT_NEAR(a, b, tol, msg) \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "FAIL: " << msg << " (expected " << (b) << " ± " << (tol) << ", got " << (a) << ")" << std::endl; \
        g_tests_failed++; \
        return false; \
    } else { \
        g_tests_passed++; \
    }

/**
 * Create test ROOT file with main tree and subframe
 */
bool CreateTestFile() {
    std::cout << "Creating test file: " << TEST_FILE << std::endl;
    
    TFile* f = TFile::Open(TEST_FILE, "RECREATE");
    if (!f || f->IsZombie()) {
        std::cerr << "Cannot create test file" << std::endl;
        return false;
    }
    
    TRandom3 rng(42);
    
    // Main tree
    TTree* mainTree = new TTree("tree", "Main tree");
    Int_t track_index;
    Float_t mX, mY;
    
    mainTree->Branch("track_index", &track_index);
    mainTree->Branch("mX", &mX);
    mainTree->Branch("mY", &mY);
    
    for (int i = 0; i < N_MAIN; i++) {
        track_index = i % N_SUB;  // Reference to subframe
        mX = rng.Gaus(0, 10);
        mY = rng.Gaus(0, 10);
        mainTree->Fill();
    }
    mainTree->Write();
    
    // Subframe T (track properties)
    TTree* subTree = new TTree("tree__subframe__T", "Subframe T");
    Float_t sub_mX, mPt, mEta;
    
    subTree->Branch("track_index", &track_index);
    subTree->Branch("mX", &sub_mX);
    subTree->Branch("mPt", &mPt);
    subTree->Branch("mEta", &mEta);
    
    for (int i = 0; i < N_SUB; i++) {
        track_index = i;
        sub_mX = rng.Gaus(0, 5);
        mPt = rng.Exp(1.0);
        mEta = rng.Gaus(0, 1);
        subTree->Fill();
    }
    subTree->Write();
    
    f->Close();
    delete f;
    
    std::cout << "  Created main tree: " << N_MAIN << " entries" << std::endl;
    std::cout << "  Created subframe T: " << N_SUB << " entries" << std::endl;
    
    return true;
}

/**
 * Test 1: LoadADFTree loads file correctly
 */
bool Test_LoadADFTree() {
    std::cout << "\n=== Test: LoadADFTree ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "LoadADFTree returns non-null");
    ASSERT_EQ(tree->GetEntries(), N_MAIN, "Main tree entry count");
    
    // Check friend was added
    TList* friends = tree->GetListOfFriends();
    ASSERT_TRUE(friends != nullptr, "Friends list exists");
    ASSERT_TRUE(friends->GetEntries() > 0, "At least one friend attached");
    
    std::cout << "  PASS: LoadADFTree" << std::endl;
    return true;
}

/**
 * Test 2: Draw main column works
 */
bool Test_DrawMainColumn() {
    std::cout << "\n=== Test: Draw Main Column ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    Int_t n = tree->Draw("mX", "", "goff");
    ASSERT_EQ(n, N_MAIN, "Draw returns correct entry count");
    
    Double_t* v = tree->GetV1();
    ASSERT_TRUE(v != nullptr, "GetV1 returns values");
    
    // Check values are reasonable (not all zero)
    double sum = 0;
    for (int i = 0; i < n; i++) sum += std::abs(v[i]);
    ASSERT_TRUE(sum > 0, "Values are non-zero");
    
    std::cout << "  PASS: Draw main column" << std::endl;
    return true;
}

/**
 * Test 3: Draw friend column with dot notation
 */
bool Test_DrawFriendColumn() {
    std::cout << "\n=== Test: Draw Friend Column (T.mX) ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // Draw subframe column via friend
    Int_t n = tree->Draw("T.mX", "", "goff");
    ASSERT_TRUE(n > 0, "Draw T.mX returns entries");
    
    Double_t* v = tree->GetV1();
    ASSERT_TRUE(v != nullptr, "GetV1 returns values");
    
    std::cout << "  Drew T.mX: " << n << " entries" << std::endl;
    std::cout << "  PASS: Draw friend column" << std::endl;
    return true;
}

/**
 * Test 4: Draw expression combining main and friend
 */
bool Test_DrawExpression() {
    std::cout << "\n=== Test: Draw Expression (mX - T.mX) ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    Int_t n = tree->Draw("mX - T.mX", "", "goff");
    ASSERT_TRUE(n > 0, "Draw expression returns entries");
    
    Double_t* v = tree->GetV1();
    ASSERT_TRUE(v != nullptr, "GetV1 returns values");
    
    // Calculate mean - should be ~0 if both are Gaussian(0, sigma)
    double mean = 0;
    for (int i = 0; i < n; i++) mean += v[i];
    mean /= n;
    
    std::cout << "  Expression mean: " << mean << std::endl;
    ASSERT_NEAR(mean, 0.0, 2.0, "Mean of difference ~0");
    
    std::cout << "  PASS: Draw expression" << std::endl;
    return true;
}

/**
 * Test 5: Draw with cut on friend column
 */
bool Test_DrawWithCut() {
    std::cout << "\n=== Test: Draw with Cut (T.mPt > 1.0) ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    Int_t n_all = tree->Draw("mX", "", "goff");
    Int_t n_cut = tree->Draw("mX", "T.mPt > 1.0", "goff");
    
    std::cout << "  All entries: " << n_all << std::endl;
    std::cout << "  After cut: " << n_cut << std::endl;
    
    ASSERT_TRUE(n_cut > 0, "Some entries pass cut");
    ASSERT_TRUE(n_cut < n_all, "Cut reduces entries");
    
    std::cout << "  PASS: Draw with cut" << std::endl;
    return true;
}

/**
 * Test 6: 2D Draw main vs friend
 */
bool Test_Draw2D() {
    std::cout << "\n=== Test: 2D Draw (mX:T.mX) ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    Int_t n = tree->Draw("mX:T.mX", "", "goff");
    ASSERT_TRUE(n > 0, "2D Draw returns entries");
    
    Double_t* vx = tree->GetV1();
    Double_t* vy = tree->GetV2();
    ASSERT_TRUE(vx != nullptr && vy != nullptr, "Both axes have values");
    
    std::cout << "  2D entries: " << n << std::endl;
    std::cout << "  PASS: 2D Draw" << std::endl;
    return true;
}

/**
 * Test 7: PrintADFBranches doesn't crash
 */
bool Test_PrintBranches() {
    std::cout << "\n=== Test: PrintADFBranches ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // Just verify it doesn't crash
    PrintADFBranches(tree);
    
    std::cout << "  PASS: PrintADFBranches" << std::endl;
    return true;
}

/**
 * Test 8: Verify numerical accuracy
 */
bool Test_NumericalAccuracy() {
    std::cout << "\n=== Test: Numerical Accuracy ===" << std::endl;
    
    // Create a simple test case with known values
    TFile* f = TFile::Open("/tmp/test_accuracy.root", "RECREATE");
    
    TTree* main = new TTree("tree", "main");
    TTree* sub = new TTree("tree__subframe__S", "sub");
    
    Int_t key;
    Float_t x, y;
    
    main->Branch("key", &key);
    main->Branch("x", &x);
    sub->Branch("key", &key);
    sub->Branch("y", &y);
    
    // Known values: x[i] = i, y[i] = 2*i, diff = -i
    for (int i = 0; i < 10; i++) {
        key = i;
        x = (Float_t)i;
        main->Fill();
        
        y = (Float_t)(2 * i);
        sub->Fill();
    }
    
    main->Write();
    sub->Write();
    f->Close();
    delete f;
    
    // Now test
    TTree* tree = LoadADFTree("/tmp/test_accuracy.root", "tree");
    ASSERT_TRUE(tree != nullptr, "Accuracy test tree loaded");
    
    Int_t n = tree->Draw("x - S.y", "", "goff");
    ASSERT_EQ(n, 10, "All entries drawn");
    
    Double_t* v = tree->GetV1();
    for (int i = 0; i < n; i++) {
        Float_t expected = -(Float_t)i;  // x - y = i - 2i = -i
        ASSERT_NEAR(v[i], expected, 0.001, TString::Format("Entry %d value", i).Data());
    }
    
    std::cout << "  PASS: Numerical accuracy" << std::endl;
    return true;
}

/**
 * Main test runner
 */
void test_AliasDataFrameTree() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "AliasDataFrameTree.C Unit Tests" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Create test data
    if (!CreateTestFile()) {
        std::cerr << "FATAL: Cannot create test file" << std::endl;
        gSystem->Exit(1);
        return;
    }
    
    // Run tests
    std::vector<bool> results;
    results.push_back(Test_LoadADFTree());
    results.push_back(Test_DrawMainColumn());
    results.push_back(Test_DrawFriendColumn());
    results.push_back(Test_DrawExpression());
    results.push_back(Test_DrawWithCut());
    results.push_back(Test_Draw2D());
    results.push_back(Test_PrintBranches());
    results.push_back(Test_NumericalAccuracy());
    
    // Summary
    int passed = 0, failed = 0;
    for (bool r : results) {
        if (r) passed++;
        else failed++;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results: " << passed << " passed, " << failed << " failed" << std::endl;
    std::cout << "Assertions: " << g_tests_passed << " passed, " << g_tests_failed << " failed" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Cleanup
    gSystem->Unlink(TEST_FILE);
    gSystem->Unlink("/tmp/test_accuracy.root");
    
    if (failed > 0) {
        std::cerr << "TESTS FAILED" << std::endl;
        gSystem->Exit(1);
    } else {
        std::cout << "ALL TESTS PASSED" << std::endl;
        gSystem->Exit(0);
    }
}
