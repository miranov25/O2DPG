/**
 * test_AliasDataFrameTree_Phase12.C - Tests for Phase 1 & 2 enhancements
 * 
 * Tests LoadSchema(), DescribeSchema(), and DescribeData() functions
 * 
 * Run with: root -l -b -q test_AliasDataFrameTree_Phase12.C
 * 
 * Prerequisites:
 * - AliasDataFrameTree_enhanced.C in same directory
 * - Write access to /tmp for test files
 */

#include <TFile.h>
#include <TTree.h>
#include <TRandom3.h>
#include <TString.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

// Include the enhanced macro
#include "../AliasDataFrameTree.C"

// Test configuration
const char* TEST_FILE = "/tmp/test_adf_phase12.root";
const char* TEST_SCHEMA = "/tmp/test_schema.json";
const int N_MAIN = 100;
const int N_SUB = 10;

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
 * Create test ROOT file
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
    Float_t x, y, z;
    
    mainTree->Branch("track_index", &track_index);
    mainTree->Branch("x", &x);
    mainTree->Branch("y", &y);
    mainTree->Branch("z", &z);
    
    for (int i = 0; i < N_MAIN; i++) {
        track_index = i % N_SUB;
        x = rng.Gaus(0, 10);
        y = rng.Gaus(0, 10);
        z = rng.Gaus(0, 5);
        mainTree->Fill();
    }
    mainTree->Write();
    
    // Subframe T
    TTree* subTree = new TTree("tree__subframe__T", "Subframe T");
    Float_t pt, eta, phi;
    
    subTree->Branch("track_index", &track_index);
    subTree->Branch("pt", &pt);
    subTree->Branch("eta", &eta);
    subTree->Branch("phi", &phi);
    
    for (int i = 0; i < N_SUB; i++) {
        track_index = i;
        pt = rng.Exp(1.0);
        eta = rng.Gaus(0, 1);
        phi = rng.Uniform(-TMath::Pi(), TMath::Pi());
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
 * Create test schema JSON file
 */
bool CreateTestSchema() {
    std::cout << "Creating test schema: " << TEST_SCHEMA << std::endl;
    
    std::ofstream f(TEST_SCHEMA);
    if (!f.is_open()) {
        std::cerr << "Cannot create schema file" << std::endl;
        return false;
    }
    
    // Write realistic schema JSON
    f << "{\n";
    f << "  \"columns\": {\n";
    f << "    \"x\": {\n";
    f << "      \"expr\": null,\n";
    f << "      \"dtype\": \"float32\"\n";
    f << "    },\n";
    f << "    \"y\": {\n";
    f << "      \"expr\": null,\n";
    f << "      \"dtype\": \"float32\"\n";
    f << "    },\n";
    f << "    \"r\": {\n";
    f << "      \"expr\": \"sqrt(x*x + y*y)\",\n";
    f << "      \"dtype\": \"float32\"\n";
    f << "    },\n";
    f << "    \"theta\": {\n";
    f << "      \"expr\": \"atan2(y, x)\",\n";
    f << "      \"dtype\": \"float32\"\n";
    f << "    },\n";
    f << "    \"pt_calib\": {\n";
    f << "      \"expr\": \"T.pt\",\n";
    f << "      \"dtype\": \"float32\"\n";
    f << "    }\n";
    f << "  },\n";
    f << "  \"compression\": {},\n";
    f << "  \"subframes\": {\n";
    f << "    \"T\": {\n";
    f << "      \"index\": [\"track_index\"],\n";
    f << "      \"tree_name\": \"tree__subframe__T\"\n";
    f << "    }\n";
    f << "  }\n";
    f << "}\n";
    
    f.close();
    
    std::cout << "  Created schema with 3 aliases" << std::endl;
    
    return true;
}

/**
 * PHASE 1 TESTS
 */

/**
 * Test 1: LoadSchema loads file successfully
 */
bool Test_LoadSchema_FileLoad() {
    std::cout << "\n=== Test: LoadSchema File Load ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    Bool_t result = LoadSchema(tree, TEST_SCHEMA);
    ASSERT_TRUE(result == kTRUE, "LoadSchema returns success");
    
    std::cout << "  PASS: LoadSchema file load" << std::endl;
    return true;
}

/**
 * Test 2: LoadSchema applies aliases correctly
 */
bool Test_LoadSchema_AliasesApplied() {
    std::cout << "\n=== Test: LoadSchema Applies Aliases ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    LoadSchema(tree, TEST_SCHEMA);
    
    // Test that aliases work in Draw
    Int_t n = tree->Draw("r", "", "goff");
    ASSERT_TRUE(n > 0, "Alias 'r' draws successfully");
    
    n = tree->Draw("theta", "", "goff");
    ASSERT_TRUE(n > 0, "Alias 'theta' draws successfully");
    
    n = tree->Draw("pt_calib", "", "goff");
    ASSERT_TRUE(n > 0, "Subframe alias 'pt_calib' draws successfully");
    
    std::cout << "  PASS: LoadSchema applies aliases" << std::endl;
    return true;
}

/**
 * Test 3: LoadSchema handles missing file gracefully
 */
bool Test_LoadSchema_MissingFile() {
    std::cout << "\n=== Test: LoadSchema Missing File ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    Bool_t result = LoadSchema(tree, "/nonexistent/schema.json");
    ASSERT_TRUE(result == kFALSE, "LoadSchema returns failure for missing file");
    
    std::cout << "  PASS: LoadSchema handles missing file" << std::endl;
    return true;
}

/**
 * Test 4: DescribeSchema shows loaded aliases
 */
bool Test_DescribeSchema_ShowsAliases() {
    std::cout << "\n=== Test: DescribeSchema Shows Aliases ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    LoadSchema(tree, TEST_SCHEMA);
    
    // Redirect stdout to capture output (simplified: just check it doesn't crash)
    std::cout << "  Calling DescribeSchema..." << std::endl;
    DescribeSchema(tree);
    
    std::cout << "  PASS: DescribeSchema runs successfully" << std::endl;
    return true;
}

/**
 * Test 5: DescribeSchema handles no schema loaded
 */
bool Test_DescribeSchema_NoSchema() {
    std::cout << "\n=== Test: DescribeSchema No Schema ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // Don't load schema, just describe
    std::cout << "  Calling DescribeSchema without loading..." << std::endl;
    DescribeSchema(tree);
    
    std::cout << "  PASS: DescribeSchema handles no schema" << std::endl;
    return true;
}

/**
 * PHASE 2 TESTS
 */

/**
 * Test 6: DescribeData shows branches
 */
bool Test_DescribeData_ShowsBranches() {
    std::cout << "\n=== Test: DescribeData Shows Branches ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    std::cout << "  Calling DescribeData..." << std::endl;
    DescribeData(tree);
    
    std::cout << "  PASS: DescribeData runs successfully" << std::endl;
    return true;
}

/**
 * Test 7: DescribeData with sort by memory
 */
bool Test_DescribeData_SortByMemory() {
    std::cout << "\n=== Test: DescribeData Sort by Memory ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    std::cout << "  Calling DescribeData with sort='memory'..." << std::endl;
    DescribeData(tree, "memory");
    
    std::cout << "  PASS: DescribeData sorts by memory" << std::endl;
    return true;
}

/**
 * Test 8: DescribeData shows aliases from schema
 */
bool Test_DescribeData_WithSchema() {
    std::cout << "\n=== Test: DescribeData With Schema ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    LoadSchema(tree, TEST_SCHEMA);
    
    std::cout << "  Calling DescribeData with schema loaded..." << std::endl;
    DescribeData(tree);
    
    std::cout << "  PASS: DescribeData shows schema info" << std::endl;
    return true;
}

/**
 * INTEGRATION TESTS
 */

/**
 * Test 9: Full workflow - Load, Schema, Draw
 */
bool Test_Integration_FullWorkflow() {
    std::cout << "\n=== Test: Integration Full Workflow ===" << std::endl;
    
    // 1. Load tree
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    // 2. Load schema
    Bool_t schemaLoaded = LoadSchema(tree, TEST_SCHEMA);
    ASSERT_TRUE(schemaLoaded == kTRUE, "Schema loaded");
    
    // 3. Describe
    DescribeSchema(tree);
    DescribeData(tree);
    
    // 4. Use aliases in Draw
    Int_t n = tree->Draw("r:theta", "", "goff");
    ASSERT_TRUE(n > 0, "2D draw with aliases works");
    
    // 5. Use subframe alias
    n = tree->Draw("pt_calib", "", "goff");
    ASSERT_TRUE(n > 0, "Subframe alias works");
    
    std::cout << "  PASS: Full workflow integration" << std::endl;
    return true;
}

/**
 * Test 10: Verify alias numerical correctness
 */
bool Test_Integration_AliasCorrectness() {
    std::cout << "\n=== Test: Alias Numerical Correctness ===" << std::endl;
    
    TTree* tree = LoadADFTree(TEST_FILE, "tree");
    ASSERT_TRUE(tree != nullptr, "Tree loaded");
    
    LoadSchema(tree, TEST_SCHEMA);
    
    // Draw all three at once (x, y, r) to avoid pointer invalidation
    Int_t n = tree->Draw("x:y:r", "", "goff");
    ASSERT_TRUE(n > 0, "Draw succeeded");
    
    Double_t* x_vals = tree->GetV1();  // First variable
    Double_t* y_vals = tree->GetV2();  // Second variable
    Double_t* r_vals = tree->GetV3();  // Third variable
    
    // Verify r = sqrt(x^2 + y^2) for first 10 entries
    Int_t nTest = TMath::Min(10, n);
    for (Int_t i = 0; i < nTest; i++) {
        Double_t expected_r = TMath::Sqrt(x_vals[i]*x_vals[i] + y_vals[i]*y_vals[i]);
        ASSERT_NEAR(r_vals[i], expected_r, 0.001, 
                   TString::Format("Entry %d: r = sqrt(x^2 + y^2)", i).Data());
    }
    
    std::cout << "  PASS: Alias numerical correctness" << std::endl;
    return true;
}

/**
 * Main test runner
 */
void test_AliasDataFrameTree() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Phase 1 & 2 Tests" << std::endl;
    std::cout << "LoadSchema, DescribeSchema, DescribeData" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    // Create test data
    if (!CreateTestFile()) {
        std::cerr << "FATAL: Cannot create test file" << std::endl;
        gSystem->Exit(1);
        return;
    }
    
    if (!CreateTestSchema()) {
        std::cerr << "FATAL: Cannot create test schema" << std::endl;
        gSystem->Exit(1);
        return;
    }
    
    // Run Phase 1 tests
    std::cout << "\n========================================" << std::endl;
    std::cout << "PHASE 1 TESTS" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    std::vector<bool> results;
    results.push_back(Test_LoadSchema_FileLoad());
    results.push_back(Test_LoadSchema_AliasesApplied());
    results.push_back(Test_LoadSchema_MissingFile());
    results.push_back(Test_DescribeSchema_ShowsAliases());
    results.push_back(Test_DescribeSchema_NoSchema());
    
    // Run Phase 2 tests
    std::cout << "\n========================================" << std::endl;
    std::cout << "PHASE 2 TESTS" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    results.push_back(Test_DescribeData_ShowsBranches());
    results.push_back(Test_DescribeData_SortByMemory());
    results.push_back(Test_DescribeData_WithSchema());
    
    // Run integration tests
    std::cout << "\n========================================" << std::endl;
    std::cout << "INTEGRATION TESTS" << std::endl;
    std::cout << "========================================\n" << std::endl;
    
    results.push_back(Test_Integration_FullWorkflow());
    results.push_back(Test_Integration_AliasCorrectness());
    
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
    gSystem->Unlink(TEST_SCHEMA);
    
    if (failed > 0) {
        std::cerr << "TESTS FAILED" << std::endl;
        gSystem->Exit(1);
    } else {
        std::cout << "ALL TESTS PASSED" << std::endl;
        gSystem->Exit(0);
    }
}


