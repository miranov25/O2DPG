/**
  .L $O2DPG/UTILS/dfextensions/AliasDataFrame/AliasDataFrameTree.C
 * AliasDataFrameTree.C - ROOT C++ macro for AliasDataFrame tree initialization
 * 
 * This macro provides helper functions to:
 * 1. Load AliasDataFrame ROOT files with subframes as friend trees
 * 2. Define aliases that work with TTree::Draw
 * 3. Handle multi-key friend tree joins
 * 
 * Usage:
 *   root -l AliasDataFrameTree.C
 *   root [0] .L AliasDataFrameTree.C
 *   root [1] auto tree = LoadADFTree("myfile.root", "tree");
 *   root [2] tree->Draw("T.mX:mX")  // Subframe T available as friend
 * 
 * Or in a macro:
 *   #include "AliasDataFrameTree.C"
 *   void myAnalysis() {
 *       auto tree = LoadADFTree("data.root", "tree");
 *       tree->Draw("mX - T.mX", "T.mPt > 1.0");
 *   }
 * 
 * Limitations:
 * - Friend trees support N:1 and 1:1 joins only (not 1:N aggregations)
 * - Multi-key joins limited to 2 keys (ROOT BuildIndex limitation)
 * - Missing keys in friend result in default values, not NaN
 */

#include <TFile.h>
#include <TTree.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <iostream>
#include <vector>
#include <map>

/**
 * Load an AliasDataFrame ROOT file and attach subframes as friend trees
 * 
 * @param filename  Path to ROOT file exported from AliasDataFrame
 * @param treename  Name of main tree (default: "tree")
 * @return          Pointer to main TTree with friends attached (caller owns)
 * 
 * Example:
 *   TTree* tree = LoadADFTree("clusters.root", "tree");
 *   tree->Draw("mX - T.mX");  // T is automatically attached as friend
 */
TTree* LoadADFTree(const char* filename, const char* treename = "tree") {
    TFile* f = TFile::Open(filename);
    if (!f || f->IsZombie()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return nullptr;
    }
    
    // Get main tree
    TTree* mainTree = (TTree*)f->Get(treename);
    if (!mainTree) {
        std::cerr << "Error: Tree '" << treename << "' not found in " << filename << std::endl;
        return nullptr;
    }
    
    // Find subframe trees (pattern: treename__subframe__NAME)
    TString prefix = TString::Format("%s__subframe__", treename);
    
    TIter nextKey(f->GetListOfKeys());
    TKey* key;
    std::vector<TString> subframeNames;
    
    while ((key = (TKey*)nextKey())) {
        TString keyName = key->GetName();
        if (keyName.BeginsWith(prefix)) {
            TString sfName = keyName(prefix.Length(), keyName.Length() - prefix.Length());
            subframeNames.push_back(sfName);
        }
    }
    
    // Attach each subframe as friend
    for (const auto& sfName : subframeNames) {
        TString sfTreeName = prefix + sfName;
        TTree* sfTree = (TTree*)f->Get(sfTreeName);
        
        if (sfTree) {
            // Try to build index - check for common index columns
            // First try single key: track_index, index, key
            bool indexed = false;
            
            for (const char* idxCol : {"track_index", "index", "key", "id"}) {
                if (sfTree->GetBranch(idxCol) && mainTree->GetBranch(idxCol)) {
                    // Check for second key (for composite index)
                    for (const char* idx2Col : {"firstTFOrbit", "firstTForbit", "orbit", "tf"}) {
                        if (sfTree->GetBranch(idx2Col) && mainTree->GetBranch(idx2Col)) {
                            sfTree->BuildIndex(idxCol, idx2Col);
                            indexed = true;
                            std::cout << "  Subframe '" << sfName << "': BuildIndex(" 
                                      << idxCol << ", " << idx2Col << ")" << std::endl;
                            break;
                        }
                    }
                    if (!indexed) {
                        sfTree->BuildIndex(idxCol);
                        indexed = true;
                        std::cout << "  Subframe '" << sfName << "': BuildIndex(" 
                                  << idxCol << ")" << std::endl;
                    }
                    break;
                }
            }
            
            if (!indexed) {
                std::cout << "  Subframe '" << sfName << "': No index (linear scan)" << std::endl;
            }
            
            // Add as friend with subframe name as alias
            mainTree->AddFriend(sfTree, sfName);
            std::cout << "  Added friend: " << sfName << std::endl;
        }
    }
    
    std::cout << "Loaded tree '" << treename << "' with " << subframeNames.size() 
              << " subframes from " << filename << std::endl;
    
    return mainTree;
}

/**
 * Load ADF tree with explicit index column specification
 * 
 * @param filename      Path to ROOT file
 * @param treename      Name of main tree
 * @param indexCols     Map of subframe name -> index column(s)
 * @return              Pointer to main TTree with friends attached
 * 
 * Example:
 *   std::map<TString, std::vector<TString>> idx;
 *   idx["T"] = {"track_index"};
 *   idx["R"] = {"index", "firstTFOrbit"};
 *   TTree* tree = LoadADFTreeWithIndex("data.root", "tree", idx);
 */
TTree* LoadADFTreeWithIndex(
    const char* filename, 
    const char* treename,
    const std::map<TString, std::vector<TString>>& indexCols
) {
    TFile* f = TFile::Open(filename);
    if (!f || f->IsZombie()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return nullptr;
    }
    
    TTree* mainTree = (TTree*)f->Get(treename);
    if (!mainTree) {
        std::cerr << "Error: Tree '" << treename << "' not found" << std::endl;
        return nullptr;
    }
    
    TString prefix = TString::Format("%s__subframe__", treename);
    
    for (const auto& [sfName, cols] : indexCols) {
        TString sfTreeName = prefix + sfName;
        TTree* sfTree = (TTree*)f->Get(sfTreeName);
        
        if (!sfTree) {
            std::cerr << "Warning: Subframe '" << sfName << "' not found" << std::endl;
            continue;
        }
        
        // Build index based on specified columns
        if (cols.size() == 1) {
            sfTree->BuildIndex(cols[0]);
            std::cout << "  " << sfName << ": BuildIndex(" << cols[0] << ")" << std::endl;
        } else if (cols.size() >= 2) {
            sfTree->BuildIndex(cols[0], cols[1]);
            std::cout << "  " << sfName << ": BuildIndex(" << cols[0] << ", " << cols[1] << ")" << std::endl;
        }
        
        mainTree->AddFriend(sfTree, sfName);
    }
    
    return mainTree;
}

/**
 * Print available branches in tree and all friends
 */
void PrintADFBranches(TTree* tree) {
    if (!tree) return;
    
    std::cout << "\n=== Main Tree: " << tree->GetName() << " ===" << std::endl;
    std::cout << "Entries: " << tree->GetEntries() << std::endl;
    std::cout << "Branches:" << std::endl;
    
    TObjArray* branches = tree->GetListOfBranches();
    for (int i = 0; i < branches->GetEntries(); i++) {
        TBranch* br = (TBranch*)branches->At(i);
        std::cout << "  " << br->GetName() << std::endl;
    }
    
    // Print friend trees
    TList* friends = tree->GetListOfFriends();
    if (friends) {
        TIter next(friends);
        TFriendElement* fe;
        while ((fe = (TFriendElement*)next())) {
            TTree* friendTree = fe->GetTree();
            std::cout << "\n=== Friend: " << fe->GetName() << " ===" << std::endl;
            std::cout << "Entries: " << friendTree->GetEntries() << std::endl;
            std::cout << "Branches (use " << fe->GetName() << ".branchname):" << std::endl;
            
            TObjArray* fBranches = friendTree->GetListOfBranches();
            for (int i = 0; i < fBranches->GetEntries(); i++) {
                TBranch* br = (TBranch*)fBranches->At(i);
                std::cout << "  " << fe->GetName() << "." << br->GetName() << std::endl;
            }
        }
    }
}

/**
 * Example usage function
 */
void ExampleUsage() {
    std::cout << R"(
=== AliasDataFrameTree.C Usage Examples ===

1. Basic loading:
   TTree* tree = LoadADFTree("clusters.root", "tree");
   tree->Draw("mX - T.mX");

2. With explicit index:
   std::map<TString, std::vector<TString>> idx;
   idx["T"] = {"track_index"};
   idx["R"] = {"index", "firstTFOrbit"};
   TTree* tree = LoadADFTreeWithIndex("data.root", "tree", idx);

3. Print structure:
   PrintADFBranches(tree);

4. Draw with cuts on friend:
   tree->Draw("mX:mY", "T.mPt > 1.0 && T.mEta < 0.5");

5. 2D plots:
   tree->Draw("mX:T.mX", "", "colz");

)" << std::endl;
}

// Auto-run example info when loaded interactively
#ifndef __CINT__
void AliasDataFrameTree() {
    std::cout << "AliasDataFrameTree.C loaded." << std::endl;
    std::cout << "Run ExampleUsage() for help." << std::endl;
}
#endif
