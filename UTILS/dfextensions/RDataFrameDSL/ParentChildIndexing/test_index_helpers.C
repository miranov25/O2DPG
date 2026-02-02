/**
 * @file test_index_helpers.C
 * @brief Basic tests for index_helpers
 * 
 * Usage:
 *   root -l -q -b 'test_index_helpers.C'
 *   
 * Or after building .so:
 *   root -l -q -b 'test_index_helpers.C("libIndexHelpers.so")'
 */

// Include header at compile time (must be in include path)
#include "index_helpers.h"
#include <iostream>
#include <cassert>
#include <cmath>

void test_index_helpers(const char* libPath = nullptr) {
    std::cout << "=== Testing index_helpers ===" << std::endl;
    
    // Load library if provided (for pre-compiled symbols)
    if (libPath) {
        std::cout << "Loading: " << libPath << std::endl;
        int loadResult = gSystem->Load(libPath);
        if (loadResult < 0) {
            std::cerr << "ERROR: Failed to load " << libPath << std::endl;
            return;
        }
    } else {
        std::cout << "Using header-only mode" << std::endl;
    }
    
    // Use fully qualified names (aliases for brevity)
    namespace IH = RDataFrameDSL::IndexHelpers;
    using RVecI = ROOT::VecOps::RVec<int>;
    using RVecF = ROOT::VecOps::RVec<float>;
    
    // ========================================
    // Test 1: ExpandParentIndex
    // ========================================
    std::cout << "\nTest 1: ExpandParentIndex" << std::endl;
    {
        RVecI firstIdx = {0, 3, 5};
        RVecI nEntries = {3, 2, 4};
        RVecI expected = {0,0,0, 1,1, 2,2,2,2};
        
        auto result = IH::ExpandParentIndex(firstIdx, nEntries);
        
        std::cout << "  Input:    firstIdx=[0,3,5], nEntries=[3,2,4]" << std::endl;
        std::cout << "  Expected: [0,0,0,1,1,2,2,2,2]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i];
            if (i < result.size()-1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); i++) {
            assert(result[i] == expected[i]);
        }
        std::cout << "  [PASS]" << std::endl;
    }
    
    // ========================================
    // Test 2: ExpandToChildren
    // ========================================
    std::cout << "\nTest 2: ExpandToChildren<float>" << std::endl;
    {
        RVecF parentValues = {1.0f, 2.0f, 3.0f};
        RVecI firstIdx = {0, 3, 5};
        RVecI nEntries = {3, 2, 4};
        RVecF expected = {1.0f,1.0f,1.0f, 2.0f,2.0f, 3.0f,3.0f,3.0f,3.0f};
        
        auto result = IH::ExpandToChildren(parentValues, firstIdx, nEntries);
        
        std::cout << "  Input:    parentValues=[1,2,3]" << std::endl;
        std::cout << "  Expected: [1,1,1,2,2,3,3,3,3]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i];
            if (i < result.size()-1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); i++) {
            assert(std::abs(result[i] - expected[i]) < 1e-6);
        }
        std::cout << "  [PASS]" << std::endl;
    }
    
    // ========================================
    // Test 3: GatherByIndex
    // ========================================
    std::cout << "\nTest 3: GatherByIndex<int>" << std::endl;
    {
        RVecI values = {100, 200, 300, 400};
        RVecI indices = {2, 0, -1, 3, 1, 99};  // -1 and 99 are invalid
        int fallback = -999;
        RVecI expected = {300, 100, -999, 400, 200, -999};
        
        auto result = IH::GatherByIndex(values, indices, fallback);
        
        std::cout << "  Input:    values=[100,200,300,400], indices=[2,0,-1,3,1,99]" << std::endl;
        std::cout << "  Expected: [300,100,-999,400,200,-999]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i];
            if (i < result.size()-1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); i++) {
            assert(result[i] == expected[i]);
        }
        std::cout << "  [PASS]" << std::endl;
    }
    
    // ========================================
    // Test 4: GetRange
    // ========================================
    std::cout << "\nTest 4: GetRange<int>" << std::endl;
    {
        RVecI values = {10, 20, 30, 40, 50, 60};
        
        // Normal case: [2,4] inclusive = [30,40,50]
        auto result1 = IH::GetRange(values, 2, 4);
        std::cout << "  GetRange([10..60], 2, 4) = [";
        for (size_t i = 0; i < result1.size(); i++) {
            std::cout << result1[i];
            if (i < result1.size()-1) std::cout << ",";
        }
        std::cout << "] (expected [30,40,50])" << std::endl;
        assert(result1.size() == 3);
        assert(result1[0] == 30 && result1[1] == 40 && result1[2] == 50);
        
        // Invalid: first=-1
        auto result2 = IH::GetRange(values, -1, 3);
        std::cout << "  GetRange([10..60], -1, 3) = [] (expected [])" << std::endl;
        assert(result2.size() == 0);
        
        // Invalid: out of bounds
        auto result3 = IH::GetRange(values, 0, 99);
        std::cout << "  GetRange([10..60], 0, 99) = [] (expected [])" << std::endl;
        assert(result3.size() == 0);
        
        std::cout << "  [PASS]" << std::endl;
    }
    
    // ========================================
    // Test 5: CountChildren
    // ========================================
    std::cout << "\nTest 5: CountChildren" << std::endl;
    {
        RVecI childToParent = {0, 0, 0, 1, 1, 2, 2, 2, 2};
        int nParents = 3;
        RVecI expected = {3, 2, 4};
        
        auto result = IH::CountChildren(childToParent, nParents);
        
        std::cout << "  Input:    childToParent=[0,0,0,1,1,2,2,2,2], nParents=3" << std::endl;
        std::cout << "  Expected: [3,2,4]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i];
            if (i < result.size()-1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); i++) {
            assert(result[i] == expected[i]);
        }
        std::cout << "  [PASS]" << std::endl;
    }
    
    // ========================================
    // Test 6: CountChildrenByCategory
    // ========================================
    std::cout << "\nTest 6: CountChildrenByCategory" << std::endl;
    {
        RVecI childToParent = {0, 0, 0, 1, 1, 2, 2, 2, 2};
        RVecI childCategory = {1, 0, 1, 1, 1, 0, 1, 0, 1};  // 0=ITS, 1=TPC
        int nParents = 3;
        int targetCategory = 1;  // Count TPC
        RVecI expected = {2, 2, 2};  // Parent 0: 2 TPC, Parent 1: 2 TPC, Parent 2: 2 TPC
        
        auto result = IH::CountChildrenByCategory(childToParent, childCategory, nParents, targetCategory);
        
        std::cout << "  Counting category=1 (TPC)" << std::endl;
        std::cout << "  Expected: [2,2,2]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i];
            if (i < result.size()-1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); i++) {
            assert(result[i] == expected[i]);
        }
        std::cout << "  [PASS]" << std::endl;
    }
    
    // ========================================
    // Test 7: ComputeEntriesFromOffsets
    // ========================================
    std::cout << "\nTest 7: ComputeEntriesFromOffsets" << std::endl;
    {
        ROOT::RVec<unsigned int> offsets = {0, 158, 309, 460};
        int totalSize = 600;
        RVecI expected = {158, 151, 151, 140};  // last = 600 - 460
        
        auto result = IH::ComputeEntriesFromOffsets(offsets, totalSize);
        
        std::cout << "  Input:    offsets=[0,158,309,460], totalSize=600" << std::endl;
        std::cout << "  Expected: [158,151,151,140]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i];
            if (i < result.size()-1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); i++) {
            assert(result[i] == expected[i]);
        }
        std::cout << "  [PASS]" << std::endl;
    }
    
    // ========================================
    // Test 8: ExpandParentIndexFromOffsets
    // ========================================
    std::cout << "\nTest 8: ExpandParentIndexFromOffsets" << std::endl;
    {
        ROOT::RVec<unsigned int> offsets = {0, 3, 5};
        int totalSize = 9;
        RVecI expected = {0,0,0, 1,1, 2,2,2,2};
        
        auto result = IH::ExpandParentIndexFromOffsets(offsets, totalSize);
        
        std::cout << "  Input:    offsets=[0,3,5], totalSize=9" << std::endl;
        std::cout << "  Expected: [0,0,0,1,1,2,2,2,2]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i];
            if (i < result.size()-1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); i++) {
            assert(result[i] == expected[i]);
        }
        std::cout << "  [PASS]" << std::endl;
    }
    
    // ========================================
    // Test 9: ExpandToChildrenFromOffsets
    // ========================================
    std::cout << "\nTest 9: ExpandToChildrenFromOffsets<float>" << std::endl;
    {
        RVecF parentValues = {1.0f, 2.0f, 3.0f};
        ROOT::RVec<unsigned int> offsets = {0, 3, 5};
        int totalSize = 9;
        RVecF expected = {1.0f,1.0f,1.0f, 2.0f,2.0f, 3.0f,3.0f,3.0f,3.0f};
        
        auto result = IH::ExpandToChildrenFromOffsets(parentValues, offsets, totalSize);
        
        std::cout << "  Input:    parentValues=[1,2,3], offsets=[0,3,5], totalSize=9" << std::endl;
        std::cout << "  Expected: [1,1,1,2,2,3,3,3,3]" << std::endl;
        std::cout << "  Got:      [";
        for (size_t i = 0; i < result.size(); i++) {
            std::cout << result[i];
            if (i < result.size()-1) std::cout << ",";
        }
        std::cout << "]" << std::endl;
        
        assert(result.size() == expected.size());
        for (size_t i = 0; i < result.size(); i++) {
            assert(std::abs(result[i] - expected[i]) < 1e-6);
        }
        std::cout << "  [PASS]" << std::endl;
    }
    
    std::cout << "\n=== All tests passed! ===" << std::endl;
}
