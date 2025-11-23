/**
  .L $O2DPG/UTILS/dfextensions/AliasDataFrame/AliasDataFrameTree.C
 * AliasDataFrameTree.C - ROOT C++ macro for AliasDataFrame tree initialization
 * 
 * This macro provides helper functions to:
 * 1. Load AliasDataFrame ROOT files with subframes as friend trees
 * 2. Load schema from JSON and apply aliases
 * 3. Describe data and schema
 * 4. Handle multi-key friend tree joins
 * 
 * Usage:
 *   root -l AliasDataFrameTree.C
 *   root [0] .L AliasDataFrameTree.C
 *   root [1] auto tree = LoadADFTree("myfile.root", "tree");
 *   root [2] LoadSchema(tree, "schema.json");  // Apply aliases from schema
 *   root [3] DescribeSchema(tree);  // Show loaded schema
 *   root [4] tree->Draw("dy:dz")  // Aliases available
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
#include <TBranch.h>
#include <TLeaf.h>
#include <TSystem.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>

// Storage for schema info (attached to tree as metadata)
struct SchemaInfo {
    std::map<TString, TString> aliases;  // name -> expression
    std::map<TString, std::vector<TString>> subframes;  // name -> index columns
    bool loaded = false;
};

// Global storage (one per tree, keyed by tree pointer)
std::map<TTree*, SchemaInfo> g_schemaRegistry;

/**
 * Simple JSON value extractor (minimal parser for our schema format)
 */
TString ExtractJSONString(const TString& json, const TString& key) {
    // Find "key": "value" pattern
    TString pattern = TString::Format("\"%s\"", key.Data());
    Int_t pos = json.Index(pattern);
    if (pos == kNPOS) return "";
    
    // Find the opening quote of value
    pos = json.Index("\"", pos + pattern.Length());
    if (pos == kNPOS) return "";
    
    // Find closing quote
    Int_t end = json.Index("\"", pos + 1);
    if (end == kNPOS) return "";
    
    return json(pos + 1, end - pos - 1);
}

/**
 * Extract array of strings from JSON
 */
std::vector<TString> ExtractJSONArray(const TString& json, const TString& key) {
    std::vector<TString> result;
    
    TString pattern = TString::Format("\"%s\"", key.Data());
    Int_t pos = json.Index(pattern);
    if (pos == kNPOS) return result;
    
    // Find the opening bracket
    pos = json.Index("[", pos);
    if (pos == kNPOS) return result;
    
    // Find closing bracket
    Int_t end = json.Index("]", pos);
    if (end == kNPOS) return result;
    
    // Extract comma-separated quoted strings
    TString arrayContent = json(pos + 1, end - pos - 1);
    TObjArray* tokens = arrayContent.Tokenize(",");
    
    for (Int_t i = 0; i < tokens->GetEntries(); i++) {
        TString token = ((TObjString*)tokens->At(i))->GetString();
        token.ReplaceAll("\"", "");
        token.ReplaceAll(" ", "");
        if (token.Length() > 0) {
            result.push_back(token);
        }
    }
    delete tokens;
    
    return result;
}

/**
 * Phase 1: Load schema from JSON file and apply to tree
 * 
 * @param tree        TTree to apply schema to
 * @param schemaPath  Path to JSON schema file (from Python export_schema)
 * @return            True if successful
 * 
 * Example:
 *   TTree* tree = LoadADFTree("data.root", "tree");
 *   LoadSchema(tree, "optimized_schema.json");
 *   tree->Draw("dy:dz");  // Aliases from schema now available
 */
Bool_t LoadSchema(TTree* tree, const char* schemaPath) {
    if (!tree) {
        std::cerr << "Error: NULL tree" << std::endl;
        return kFALSE;
    }
    
    // Read entire JSON file
    std::ifstream file(schemaPath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open schema file: " << schemaPath << std::endl;
        return kFALSE;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    TString json = buffer.str();
    file.close();
    
    std::cout << "Loading schema from: " << schemaPath << std::endl;
    
    // Initialize schema info for this tree
    SchemaInfo& schema = g_schemaRegistry[tree];
    schema.aliases.clear();
    schema.subframes.clear();
    
    // Parse columns section for aliases
    // Format: "columns": { "name": { "expr": "expression", ...}, ...}
    Int_t colStart = json.Index("\"columns\"");
    if (colStart == kNPOS) {
        std::cerr << "Warning: No 'columns' section in schema" << std::endl;
    } else {
        // Find the columns object
        Int_t objStart = json.Index("{", colStart);
        Int_t depth = 0;
        Int_t objEnd = objStart;
        
        // Simple brace matching to find end of columns object
        for (Int_t i = objStart; i < json.Length(); i++) {
            if (json[i] == '{') depth++;
            if (json[i] == '}') {
                depth--;
                if (depth == 0) {
                    objEnd = i;
                    break;
                }
            }
        }
        
        TString columnsSection = json(objStart, objEnd - objStart + 1);
        
        // Extract each column definition (simplified parsing)
        // Look for "columnName": { "expr": "expression" }
        TObjArray* lines = columnsSection.Tokenize("\n");
        TString currentCol = "";
        
        for (Int_t i = 0; i < lines->GetEntries(); i++) {
            TString line = ((TObjString*)lines->At(i))->GetString();
            line.ReplaceAll(" ", "");
            line.ReplaceAll("\t", "");
            
            // Column name line: "colName":{
            if (line.Contains("\":{")) {
                Int_t qStart = line.Index("\"");
                Int_t qEnd = line.Index("\"", qStart + 1);
                if (qStart != kNPOS && qEnd != kNPOS) {
                    currentCol = line(qStart + 1, qEnd - qStart - 1);
                }
            }
            // Expression line: "expr":"something"
            else if (line.Contains("\"expr\"") && currentCol.Length() > 0) {
                TString expr = ExtractJSONString(line, "expr");
                if (expr.Length() > 0 && expr != "null") {
                    schema.aliases[currentCol] = expr;
                    tree->SetAlias(currentCol, expr);
                }
                currentCol = "";  // Reset
            }
        }
        delete lines;
    }
    
    // Parse subframes section
    Int_t sfStart = json.Index("\"subframes\"");
    if (sfStart != kNPOS) {
        Int_t objStart = json.Index("{", sfStart);
        Int_t depth = 0;
        Int_t objEnd = objStart;
        
        for (Int_t i = objStart; i < json.Length(); i++) {
            if (json[i] == '{') depth++;
            if (json[i] == '}') {
                depth--;
                if (depth == 0) {
                    objEnd = i;
                    break;
                }
            }
        }
        
        TString subframesSection = json(objStart, objEnd - objStart + 1);
        
        // Extract subframe names and index columns
        TObjArray* lines = subframesSection.Tokenize("\n");
        TString currentSF = "";
        
        for (Int_t i = 0; i < lines->GetEntries(); i++) {
            TString line = ((TObjString*)lines->At(i))->GetString();
            line.ReplaceAll(" ", "");
            line.ReplaceAll("\t", "");
            
            if (line.Contains("\":{")) {
                Int_t qStart = line.Index("\"");
                Int_t qEnd = line.Index("\"", qStart + 1);
                if (qStart != kNPOS && qEnd != kNPOS) {
                    currentSF = line(qStart + 1, qEnd - qStart - 1);
                }
            }
            else if (line.Contains("\"index\"") && currentSF.Length() > 0) {
                std::vector<TString> indexCols = ExtractJSONArray(line, "index");
                if (indexCols.size() > 0) {
                    schema.subframes[currentSF] = indexCols;
                }
                currentSF = "";
            }
        }
        delete lines;
    }
    
    schema.loaded = kTRUE;
    
    std::cout << "  Loaded " << schema.aliases.size() << " aliases" << std::endl;
    std::cout << "  Found " << schema.subframes.size() << " subframe definitions" << std::endl;
    
    return kTRUE;
}

/**
 * Phase 1: Describe loaded schema
 * 
 * @param tree  TTree with loaded schema
 * 
 * Example:
 *   DescribeSchema(tree);
 */
void DescribeSchema(TTree* tree) {
    if (!tree) {
        std::cerr << "Error: NULL tree" << std::endl;
        return;
    }
    
    auto it = g_schemaRegistry.find(tree);
    if (it == g_schemaRegistry.end() || !it->second.loaded) {
        std::cout << "No schema loaded for this tree" << std::endl;
        std::cout << "Use LoadSchema(tree, \"path/to/schema.json\") first" << std::endl;
        return;
    }
    
    const SchemaInfo& schema = it->second;
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Schema Overview" << std::endl;
    std::cout << "========================================" << std::endl;
    
    std::cout << "\nAliases: " << schema.aliases.size() << std::endl;
    if (schema.aliases.size() > 0) {
        std::cout << "Name                 Expression" << std::endl;
        std::cout << "-------------------- ------------------------------------" << std::endl;
        for (const auto& [name, expr] : schema.aliases) {
            printf("%-20s %s\n", name.Data(), expr.Data());
        }
    }
    
    std::cout << "\nSubframes: " << schema.subframes.size() << std::endl;
    if (schema.subframes.size() > 0) {
        std::cout << "Name                 Index Columns" << std::endl;
        std::cout << "-------------------- ------------------------------------" << std::endl;
        for (const auto& [name, cols] : schema.subframes) {
            TString colList = "";
            for (size_t i = 0; i < cols.size(); i++) {
                if (i > 0) colList += ", ";
                colList += cols[i];
            }
            printf("%-20s [%s]\n", name.Data(), colList.Data());
        }
    }
    
    std::cout << "========================================\n" << std::endl;
}

/**
 * Phase 2: Describe data in tree (branches and memory usage)
 * 
 * @param tree      TTree to describe
 * @param sortBy    Sort order: "name", "memory", "type" (default: "name")
 * 
 * Example:
 *   DescribeData(tree, "memory");  // Sort by memory usage
 */
void DescribeData(TTree* tree, const char* sortBy = "name") {
    if (!tree) {
        std::cerr << "Error: NULL tree" << std::endl;
        return;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Data Description: " << tree->GetName() << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Entries: " << tree->GetEntries() << std::endl;
    
    // Collect branch info
    struct BranchInfo {
        TString name;
        TString type;
        Long64_t bytes;
        Double_t memory_mb;
    };
    std::vector<BranchInfo> branches;
    
    TObjArray* branchList = tree->GetListOfBranches();
    Long64_t totalBytes = 0;
    
    for (Int_t i = 0; i < branchList->GetEntries(); i++) {
        TBranch* br = (TBranch*)branchList->At(i);
        BranchInfo info;
        info.name = br->GetName();
        
        // Get type
        TLeaf* leaf = (TLeaf*)br->GetListOfLeaves()->At(0);
        if (leaf) {
            info.type = leaf->GetTypeName();
        } else {
            info.type = "unknown";
        }
        
        // Get memory usage
        info.bytes = br->GetTotBytes();
        info.memory_mb = info.bytes / (1024.0 * 1024.0);
        totalBytes += info.bytes;
        
        branches.push_back(info);
    }
    
    // Sort
    TString sortMode = sortBy;
    if (sortMode == "memory") {
        std::sort(branches.begin(), branches.end(),
                  [](const BranchInfo& a, const BranchInfo& b) {
                      return a.bytes > b.bytes;
                  });
    } else if (sortMode == "type") {
        std::sort(branches.begin(), branches.end(),
                  [](const BranchInfo& a, const BranchInfo& b) {
                      return a.type < b.type;
                  });
    }
    // else: already in name order
    
    // Print table
    std::cout << "\nPhysical Columns:" << std::endl;
    printf("%-20s %-12s %12s\n", "Name", "Type", "Memory (MB)");
    std::cout << "-------------------- ------------ ------------" << std::endl;
    
    for (const auto& info : branches) {
        printf("%-20s %-12s %12.2f\n", 
               info.name.Data(), 
               info.type.Data(), 
               info.memory_mb);
    }
    
    std::cout << "-------------------- ------------ ------------" << std::endl;
    printf("%-20s %-12s %12.2f\n", "TOTAL", "", totalBytes / (1024.0 * 1024.0));
    
    // Show aliases if schema loaded
    auto it = g_schemaRegistry.find(tree);
    if (it != g_schemaRegistry.end() && it->second.loaded) {
        std::cout << "\nComputed Columns (Aliases): " << it->second.aliases.size() << std::endl;
        if (it->second.aliases.size() > 0) {
            printf("%-20s %s\n", "Name", "Expression");
            std::cout << "-------------------- ------------------------------------" << std::endl;
            for (const auto& [name, expr] : it->second.aliases) {
                printf("%-20s %s\n", name.Data(), expr.Data());
            }
        }
    }
    
    // Show friend trees
    TList* friends = tree->GetListOfFriends();
    if (friends && friends->GetEntries() > 0) {
        std::cout << "\nFriend Trees (Subframes): " << friends->GetEntries() << std::endl;
        TIter next(friends);
        TFriendElement* fe;
        while ((fe = (TFriendElement*)next())) {
            TTree* ft = fe->GetTree();
            printf("  %-20s %10lld entries\n", fe->GetName(), ft->GetEntries());
        }
    }
    
    std::cout << "========================================\n" << std::endl;
}

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

2. Load with schema:
   TTree* tree = LoadADFTree("data.root", "tree");
   LoadSchema(tree, "optimized_schema.json");
   tree->Draw("dy:dz");  // Aliases from schema

3. Describe schema:
   DescribeSchema(tree);

4. Describe data:
   DescribeData(tree);
   DescribeData(tree, "memory");  // Sort by memory

5. With explicit index:
   std::map<TString, std::vector<TString>> idx;
   idx["T"] = {"track_index"};
   idx["R"] = {"index", "firstTFOrbit"};
   TTree* tree = LoadADFTreeWithIndex("data.root", "tree", idx);

6. Draw with cuts on friend:
   tree->Draw("mX:mY", "T.mPt > 1.0 && T.mEta < 0.5");

)" << std::endl;
}

// Auto-run example info when loaded interactively
#ifndef __CINT__
void AliasDataFrameTree() {
    std::cout << "AliasDataFrameTree.C loaded (Phase 1 + 2 enhanced)." << std::endl;
    std::cout << "New features: LoadSchema(), DescribeSchema(), DescribeData()" << std::endl;
    std::cout << "Run ExampleUsage() for help." << std::endl;
}
#endif
