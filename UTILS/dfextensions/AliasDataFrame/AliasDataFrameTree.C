/**
  .L $O2DPG/UTILS/dfextensions/AliasDataFrame/AliasDataFrameTree.C
 * AliasDataFrameTree.C - ROOT C++ macro for AliasDataFrame tree initialization
 * 
 * This macro provides helper functions to:
 * 1. Load AliasDataFrame ROOT files with subframes as friend trees
 * 2. Load schema from JSON or embedded ADF_SCHEMA and apply aliases
 * 3. Build N-key composite indices (overcoming ROOT's 2-key limit)
 * 4. Describe data and schema
 * 
 * Phase 3.1: In-memory composite index for read-only files
 * - Uses TMemFile to clone subframes when file is read-only
 * - Full TTree::Draw compatibility maintained
 * 
 * Usage:
 *   root -l AliasDataFrameTree.C
 *   root [0] .L AliasDataFrameTree.C
 *   root [1] auto tree = LoadADFTree("myfile.root", "tree");
 *   root [2] tree->Draw("dy:dz")  // Aliases and N-key indices work automatically
 * 
 * For files with embedded schema (Python save_schema_to_root):
 *   - Schema is auto-loaded
 *   - Multi-key indices (>2 columns) use composite key
 * 
 * Limitations:
 * - Friend trees support N:1 and 1:1 joins only (not 1:N aggregations)
 * - Missing keys in friend result in default values, not NaN
 */

#include <TFile.h>
#include <TMemFile.h>
#include <TTree.h>
#include <TKey.h>
#include <TString.h>
#include <TObjArray.h>
#include <TObjString.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TSystem.h>
#include <TList.h>
#include <TFriendElement.h>
#include <TTreeFormula.h>
#include <TStopwatch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>

// ============================================================================
// Schema Storage
// ============================================================================

struct SchemaInfo {
    std::map<TString, TString> aliases;  // name -> expression
    std::map<TString, std::vector<TString>> subframeIndices;  // subframe -> index columns
    bool loaded = false;
};

// Global storage (one per tree, keyed by tree pointer)
std::map<TTree*, SchemaInfo> g_schemaRegistry;

// Storage for in-memory files (keeps them alive for session)
std::vector<TMemFile*> g_memFiles;

// Forward declarations
Bool_t LoadSchemaFromJSON(TTree* tree, const TString& json);

// ============================================================================
// JSON Parsing Utilities
// ============================================================================

/**
 * Simple JSON value extractor for string values
 */
TString ExtractJSONString(const TString& json, const TString& key) {
    TString pattern = TString::Format("\"%s\"", key.Data());
    Ssiz_t pos = json.Index(pattern);
    if (pos == kNPOS) return "";
    
    pos = json.Index("\"", pos + pattern.Length());
    if (pos == kNPOS) return "";
    
    Ssiz_t end = json.Index("\"", pos + 1);
    if (end == kNPOS) return "";
    
    return json(pos + 1, end - pos - 1);
}

/**
 * Extract array of strings from JSON (for index columns)
 */
std::vector<TString> ExtractJSONArray(const TString& json, const TString& key) {
    std::vector<TString> result;
    
    TString pattern = TString::Format("\"%s\"", key.Data());
    Ssiz_t pos = json.Index(pattern);
    if (pos == kNPOS) return result;
    
    pos = json.Index("[", pos);
    if (pos == kNPOS) {
        // Might be a single string value instead of array
        TString singleVal = ExtractJSONString(json.Data() + pos, key);
        if (singleVal.Length() > 0) {
            result.push_back(singleVal);
        }
        return result;
    }
    
    Ssiz_t end = json.Index("]", pos);
    if (end == kNPOS) return result;
    
    TString arrayContent = json(pos + 1, end - pos - 1);
    TObjArray* tokens = arrayContent.Tokenize(",");
    
    for (Int_t i = 0; i < tokens->GetEntries(); i++) {
        TString token = ((TObjString*)tokens->At(i))->GetString();
        token.ReplaceAll("\"", "");
        token.ReplaceAll(" ", "");
        token.ReplaceAll("\n", "");
        token.ReplaceAll("\t", "");
        if (token.Length() > 0) {
            result.push_back(token);
        }
    }
    delete tokens;
    
    return result;
}

/**
 * Extract a subsection of JSON (e.g., "subframes": {...})
 */
TString ExtractJSONObject(const TString& json, const TString& key) {
    TString pattern = TString::Format("\"%s\"", key.Data());
    Ssiz_t pos = json.Index(pattern);
    if (pos == kNPOS) return "";
    
    Ssiz_t objStart = json.Index("{", pos);
    if (objStart == kNPOS) return "";
    
    Int_t depth = 0;
    Ssiz_t objEnd = objStart;
    
    for (Ssiz_t i = objStart; i < json.Length(); i++) {
        if (json[i] == '{') depth++;
        if (json[i] == '}') {
            depth--;
            if (depth == 0) {
                objEnd = i;
                break;
            }
        }
    }
    
    return json(objStart, objEnd - objStart + 1);
}

// ============================================================================
// In-Memory Subframe Cloning (Phase 3.1)
// ============================================================================

/**
 * Clone a subframe tree into an in-memory TMemFile.
 * This allows adding branches for composite index even when original file is read-only.
 * 
 * @param subframe   Original subframe tree (read-only)
 * @param name       Name for the cloned tree
 * @return           Cloned tree in memory (writable)
 */
TTree* CloneSubframeToMemory(TTree* subframe, const TString& name) {
    // Create in-memory file with unique name
    TString memFileName = TString::Format("adf_mem_%s_%p", name.Data(), (void*)subframe);
    TMemFile* memFile = new TMemFile(memFileName.Data(), "RECREATE");
    g_memFiles.push_back(memFile);  // Keep alive for session lifetime
    
    memFile->cd();
    
    // Clone the entire tree (structure + data)
    // "fast" option uses basket-by-basket copy (most efficient)
    TTree* clone = subframe->CloneTree(-1, "fast");
    clone->SetName(name.Data());
    clone->SetDirectory(memFile);
    
    return clone;
}

/**
 * Check if a TFile is writable
 */
Bool_t IsFileWritable(TFile* file) {
    if (!file) return kFALSE;
    return file->IsWritable();
}

// ============================================================================
// Composite Index Implementation (Phase 3)
// ============================================================================

/**
 * Check if a TLeaf represents an integer type
 */
Bool_t IsIntegerLeaf(TLeaf* leaf) {
    if (!leaf) return kFALSE;
    
    TString typeName = leaf->GetTypeName();
    return (typeName == "Int_t" || typeName == "UInt_t" ||
            typeName == "Short_t" || typeName == "UShort_t" ||
            typeName == "Long_t" || typeName == "ULong_t" ||
            typeName == "Long64_t" || typeName == "ULong64_t" ||
            typeName == "Char_t" || typeName == "UChar_t" ||
            typeName == "int" || typeName == "unsigned int" ||
            typeName == "short" || typeName == "unsigned short" ||
            typeName == "long" || typeName == "unsigned long" ||
            typeName == "long long" || typeName == "unsigned long long");
}

/**
 * Build composite index for subframe with N > 2 index columns.
 * 
 * Uses CARDINALITY-BASED PACKING:
 * - Maps each column's values to compact codes [0, 1, 2, ...]
 * - Uses cardinality (not max+1) as base
 * - Handles sparse indices (e.g., firstTFOrbit) correctly
 * - Collision-free within the indexed tree
 * 
 * For read-only files:
 * - Clones subframe to TMemFile
 * - Builds composite key on clone
 * - Returns clone as the tree to use as friend
 * 
 * @param mainTree       Main tree (used to create matching key column)
 * @param subframeTree   Subframe TTree to index (will be cloned if read-only)
 * @param columns        Index column names from schema
 * @param subframeName   Name of subframe (for unique branch naming)
 * @param outTree        [output] Tree to use as friend (may be clone)
 * @return               true if successful, false otherwise
 */
Bool_t BuildCompositeIndex(TTree* mainTree, TTree* subframeTree, 
                           const std::vector<TString>& columns,
                           const TString& subframeName,
                           TTree*& outTree) {
    outTree = subframeTree;  // Default: use original
    
    if (columns.size() <= 2) {
        // Use native ROOT BuildIndex for 1-2 keys
        if (columns.size() == 1) {
            // ROOT friend tree lookup: when iterating the main tree, ROOT
            // evaluates the friend's index formula on the main tree values.
            // BuildIndex(major, minor) stores the index as major*multiplier + minor.
            // 
            // BUG: BuildIndex("gid") with no minor uses "" as minor formula.
            // On the friend tree (which has "gid"), minor evaluates to 0 → stored as gid*1 + 0.
            // But when ROOT evaluates "" on the main tree for lookup, it can fail
            // to match because the empty formula may not evaluate to exactly 0.
            //
            // FIX: Explicitly pass "0" as minor so both friend storage and main
            // tree lookup use the same (gid, 0) pair.
            subframeTree->BuildIndex(columns[0].Data(), "0");
            std::cout << "    BuildIndex(" << columns[0] << ", 0)  [single-key with explicit minor]" << std::endl;
        } else if (columns.size() == 2) {
            subframeTree->BuildIndex(columns[0].Data(), columns[1].Data());
            std::cout << "    BuildIndex(" << columns[0] << ", " << columns[1] << ")" << std::endl;
        }
        return kTRUE;
    }
    
    // N > 2 keys: need composite index
    // Check if file is writable
    TFile* file = subframeTree->GetCurrentFile();
    Bool_t isWritable = IsFileWritable(file);
    
    TTree* workTree = subframeTree;
    
    if (!isWritable) {
        // Clone subframe to memory for index building
        std::cout << "    File is read-only, cloning subframe to memory..." << std::endl;
        workTree = CloneSubframeToMemory(subframeTree, subframeName);
        outTree = workTree;  // Return clone as the friend tree
        std::cout << "    Cloned " << workTree->GetEntries() << " entries to memory" << std::endl;
    }
    
    Long64_t nEntries = workTree->GetEntries();
    size_t nCols = columns.size();
    
    std::cout << "    BuildCompositeIndex: " << nCols << " keys, " 
              << nEntries << " entries" << std::endl;
    
    // Verify all columns exist and are integer types
    std::vector<TLeaf*> leaves(nCols);
    for (size_t i = 0; i < nCols; i++) {
        TLeaf* leaf = workTree->GetLeaf(columns[i].Data());
        if (!leaf) {
            std::cerr << "ERROR: Column '" << columns[i] << "' not found in subframe" << std::endl;
            return kFALSE;
        }
        if (!IsIntegerLeaf(leaf)) {
            std::cerr << "ERROR: Column '" << columns[i] << "' is not integer type ("
                      << leaf->GetTypeName() << "). Composite index requires integer columns." << std::endl;
            return kFALSE;
        }
        leaves[i] = leaf;
    }
    
    TStopwatch timer;
    
    // =========================================================
    // PASS 1: Build value → code dictionaries for each column
    // =========================================================
    timer.Start();
    std::vector<std::unordered_map<Long64_t, Long64_t>> valueToCodes(nCols);
    
    for (Long64_t entry = 0; entry < nEntries; entry++) {
        workTree->GetEntry(entry);
        for (size_t i = 0; i < nCols; i++) {
            Long64_t value = (Long64_t)leaves[i]->GetValue();
            auto& dict = valueToCodes[i];
            if (dict.find(value) == dict.end()) {
                dict[value] = dict.size();  // Assign next code: 0, 1, 2, ...
            }
        }
    }
    timer.Stop();
    std::cout << "      Pass 1 (build dictionaries): " << timer.RealTime() << " s" << std::endl;
    
    // Report cardinalities and check for warnings
    std::vector<Long64_t> cardinalities(nCols);
    for (size_t i = 0; i < nCols; i++) {
        cardinalities[i] = valueToCodes[i].size();
        std::cout << "      " << columns[i] << ": " << cardinalities[i] 
                  << " distinct values" << std::endl;
        
        // Warn if cardinality is very high
        if (cardinalities[i] > 1000000) {
            std::cerr << "WARNING: Column '" << columns[i] << "' has " 
                      << cardinalities[i] << " distinct values. "
                      << "This may use significant memory." << std::endl;
        }
    }
    
    // =========================================================
    // Check for overflow BEFORE building
    // =========================================================
    Long64_t keySpace = 1;
    for (size_t i = 0; i < nCols; i++) {
        // Check if multiplication would overflow
        if (keySpace > (1LL << 62) / cardinalities[i]) {
            std::cerr << "ERROR: Composite key space exceeds Long64_t limit" << std::endl;
            std::cerr << "  Total combinations would overflow: " << keySpace 
                      << " * " << cardinalities[i] << std::endl;
            std::cerr << "  Subframe will not be indexed (linear scan fallback)" << std::endl;
            return kFALSE;
        }
        keySpace *= cardinalities[i];
    }
    std::cout << "      Key space: " << keySpace << " combinations (safe)" << std::endl;
    
    // =========================================================
    // PASS 2: Create composite key branch in SUBFRAME (or clone)
    // =========================================================
    timer.Start();
    TString keyBranchName = TString::Format("__adf_key_%s__", 
        subframeName.Length() > 0 ? subframeName.Data() : "idx");
    
    Long64_t compositeKey;
    TBranch* keyBranch = workTree->Branch(keyBranchName.Data(), &compositeKey, 
                                          TString::Format("%s/L", keyBranchName.Data()).Data());
    
    for (Long64_t entry = 0; entry < nEntries; entry++) {
        workTree->GetEntry(entry);
        
        // Pack codes using cardinality as base
        compositeKey = 0;
        Long64_t multiplier = 1;
        for (size_t i = 0; i < nCols; i++) {
            Long64_t value = (Long64_t)leaves[i]->GetValue();
            Long64_t code = valueToCodes[i][value];
            compositeKey += code * multiplier;
            multiplier *= cardinalities[i];
        }
        
        keyBranch->Fill();
    }
    timer.Stop();
    std::cout << "      Pass 2 (subframe keys): " << timer.RealTime() << " s" << std::endl;
    
    // Build index on subframe's composite key
    workTree->BuildIndex(keyBranchName.Data());
    
    // =========================================================
    // Create matching composite key column in MAIN TREE
    // (Required for ROOT friend tree join to work)
    // Uses TTreeFormula to handle aliases as well as branches
    // =========================================================
    TFile* mainFile = mainTree->GetCurrentFile();
    Bool_t mainWritable = IsFileWritable(mainFile);
    
    // =========================================================
    // Handle case where index columns might be aliases (not branches)
    // Use TTreeFormula to evaluate - works for both branches and aliases
    // =========================================================
    Long64_t mainEntries = mainTree->GetEntries();
    
    // Create TTreeFormulas for each index column (handles aliases)
    std::vector<TTreeFormula*> mainFormulas(nCols);
    Bool_t allFormulasValid = kTRUE;
    
    for (size_t i = 0; i < nCols; i++) {
        TString formulaName = TString::Format("idx_%s_%zu", subframeName.Data(), i);
        mainFormulas[i] = new TTreeFormula(formulaName.Data(), columns[i].Data(), mainTree);
        
        if (mainFormulas[i]->GetNdim() == 0) {
            std::cerr << "ERROR: Cannot evaluate '" << columns[i] 
                      << "' in main tree (not a branch or valid alias)" << std::endl;
            allFormulasValid = kFALSE;
        }
    }
    
    if (!allFormulasValid) {
        // Cleanup formulas
        for (auto* f : mainFormulas) if (f) delete f;
        std::cerr << "    Composite index FAILED for main tree" << std::endl;
        return kFALSE;
    }
    
    if (mainWritable) {
        // Writable: add branch directly to main tree
        
        Long64_t mainCompositeKey;
        TBranch* mainKeyBranch = mainTree->Branch(keyBranchName.Data(), &mainCompositeKey,
                                                   TString::Format("%s/L", keyBranchName.Data()).Data());
        
        // =====================================================
        // FAST PATH: Use TTree::Draw to read all values at once
        // =====================================================
        timer.Start();
        
        mainTree->SetEstimate(mainEntries);
        
        TString drawExpr = columns[0];
        for (size_t i = 1; i < nCols; i++) {
            drawExpr += ":" + columns[i];
        }
        
        std::cout << "    Using fast Draw() path: " << drawExpr << std::endl;
        
        Long64_t nRead = mainTree->Draw(drawExpr.Data(), "", "goff");
        timer.Stop();
        std::cout << "      Draw():      " << timer.RealTime() << " s (" << nRead << " entries)" << std::endl;
        
        // Get value arrays using GetVal(i) - supports any number of columns
        std::vector<Double_t*> valueArrays(nCols);
        for (size_t i = 0; i < nCols; i++) {
            valueArrays[i] = mainTree->GetVal(i);
            if (!valueArrays[i]) {
                std::cerr << "ERROR: GetVal(" << i << ") returned null" << std::endl;
                return kFALSE;
            }
        }
        
        // =====================================================
        // OPTIMIZED: Pre-compute all keys, then bulk fill
        // =====================================================
        timer.Start();
        
        // Step 1: Pre-convert value arrays to code arrays
        std::vector<std::vector<Long64_t>> codeArrays(nCols);
        for (size_t i = 0; i < nCols; i++) {
            codeArrays[i].resize(nRead);
            Double_t* values = valueArrays[i];
            auto& dict = valueToCodes[i];
            
            for (Long64_t entry = 0; entry < nRead; entry++) {
                Long64_t value = (Long64_t)values[entry];
                auto it = dict.find(value);
                codeArrays[i][entry] = (it != dict.end()) ? it->second : -1;
            }
        }
        timer.Stop();
        std::cout << "      Codes:       " << timer.RealTime() << " s" << std::endl;
        
        // Step 2: Compute all composite keys
        timer.Start();
        std::vector<Long64_t> allKeys(nRead);
        
        for (Long64_t entry = 0; entry < nRead; entry++) {
            Long64_t key = 0;
            Long64_t multiplier = 1;
            
            for (size_t i = 0; i < nCols; i++) {
                Long64_t code = codeArrays[i][entry];
                if (code == -1) {
                    key = -1;
                    break;
                }
                key += code * multiplier;
                multiplier *= cardinalities[i];
            }
            allKeys[entry] = key;
        }
        timer.Stop();
        std::cout << "      Keys:        " << timer.RealTime() << " s" << std::endl;
        
        // Step 3: Bulk fill
        timer.Start();
        for (Long64_t entry = 0; entry < nRead; entry++) {
            mainCompositeKey = allKeys[entry];
            mainKeyBranch->Fill();
        }
        timer.Stop();
        std::cout << "      Fill:        " << timer.RealTime() << " s" << std::endl;
        
        std::cout << "    Added composite key branch to main tree" << std::endl;
    } else {
        // Read-only: create auxiliary key tree in memory
        std::cout << "    Main tree read-only, creating auxiliary key tree in memory..." << std::endl;
        
        TString memFileName = TString::Format("adf_mainkey_%s_%p", subframeName.Data(), (void*)mainTree);
        TMemFile* memFile = new TMemFile(memFileName.Data(), "RECREATE");
        g_memFiles.push_back(memFile);
        
        memFile->cd();
        
        // Create tree with same number of entries as main tree
        // Tree name must be unique per subframe to avoid conflicts
        TString auxTreeName = TString::Format("__adf_aux_%s__", subframeName.Data());
        TTree* auxKeyTree = new TTree(auxTreeName.Data(), "Auxiliary composite key");
        
        Long64_t mainCompositeKey;
        auxKeyTree->Branch(keyBranchName.Data(), &mainCompositeKey,
                          TString::Format("%s/L", keyBranchName.Data()).Data());
        
        // =====================================================
        // FAST PATH: Use TTree::Draw to read all values at once
        // This is ~10x faster than TTreeFormula per-entry loop
        // =====================================================
        timer.Start();
        
        // Set estimate to handle all entries (default is 1M)
        mainTree->SetEstimate(mainEntries);
        
        // Build draw expression: "col0:col1:col2"
        TString drawExpr = columns[0];
        for (size_t i = 1; i < nCols; i++) {
            drawExpr += ":" + columns[i];
        }
        
        std::cout << "    Using fast Draw() path: " << drawExpr << std::endl;
        
        // Read all values at once (much faster than per-entry TTreeFormula)
        Long64_t nRead = mainTree->Draw(drawExpr.Data(), "", "goff");
        timer.Stop();
        std::cout << "      Draw():      " << timer.RealTime() << " s (" << nRead << " entries)" << std::endl;
        
        if (nRead != mainEntries) {
            std::cerr << "WARNING: Draw returned " << nRead << " entries, expected " << mainEntries << std::endl;
        }
        
        // Get value arrays using GetVal(i) - supports any number of columns
        std::vector<Double_t*> valueArrays(nCols);
        for (size_t i = 0; i < nCols; i++) {
            valueArrays[i] = mainTree->GetVal(i);
            if (!valueArrays[i]) {
                std::cerr << "ERROR: GetVal(" << i << ") returned null" << std::endl;
                return kFALSE;
            }
        }
        
        // =====================================================
        // OPTIMIZED: Pre-compute all keys, then bulk fill
        // =====================================================
        timer.Start();
        
        // Step 1: Pre-convert value arrays to code arrays (eliminates hash lookups from hot loop)
        std::vector<std::vector<Long64_t>> codeArrays(nCols);
        for (size_t i = 0; i < nCols; i++) {
            codeArrays[i].resize(nRead);
            Double_t* values = valueArrays[i];
            auto& dict = valueToCodes[i];
            
            for (Long64_t entry = 0; entry < nRead; entry++) {
                Long64_t value = (Long64_t)values[entry];
                auto it = dict.find(value);
                codeArrays[i][entry] = (it != dict.end()) ? it->second : -1;
            }
        }
        timer.Stop();
        std::cout << "      Codes:       " << timer.RealTime() << " s" << std::endl;
        
        // Step 2: Compute all composite keys (pure arithmetic, no lookups)
        timer.Start();
        std::vector<Long64_t> allKeys(nRead);
        
        for (Long64_t entry = 0; entry < nRead; entry++) {
            Long64_t key = 0;
            Long64_t multiplier = 1;
            Bool_t valid = kTRUE;
            
            for (size_t i = 0; i < nCols; i++) {
                Long64_t code = codeArrays[i][entry];
                if (code == -1) {
                    key = -1;
                    valid = kFALSE;
                    break;
                }
                key += code * multiplier;
                multiplier *= cardinalities[i];
            }
            allKeys[entry] = key;
        }
        timer.Stop();
        std::cout << "      Keys:        " << timer.RealTime() << " s" << std::endl;
        
        // Step 3: Bulk fill tree (optimized settings)
        timer.Start();
        auxKeyTree->SetAutoFlush(0);  // Disable auto-flush during bulk fill
        
        for (Long64_t entry = 0; entry < nRead; entry++) {
            mainCompositeKey = allKeys[entry];
            auxKeyTree->Fill();
        }
        
        auxKeyTree->FlushBaskets();  // Single flush at end
        timer.Stop();
        std::cout << "      Fill:        " << timer.RealTime() << " s" << std::endl;
        
        // Add auxiliary tree as friend to main tree
        // This makes __adf_key_SF__ available for TTreeFormula
        mainTree->AddFriend(auxKeyTree);
        
        std::cout << "    Created auxiliary key tree: " << mainEntries 
                  << " entries, " << (mainEntries * 8 / 1024.0 / 1024.0) 
                  << " MB" << std::endl;
    }
    
    // Cleanup formulas
    for (auto* f : mainFormulas) delete f;
    
    // NOTE: Do NOT call BuildIndex on main tree!
    // ROOT will use the subframe's index when evaluating friend expressions
    
    std::cout << "    Composite index built successfully" << std::endl;
    return kTRUE;
}

// Overload for backward compatibility (without outTree parameter)
Bool_t BuildCompositeIndex(TTree* mainTree, TTree* subframeTree, 
                           const std::vector<TString>& columns,
                           const TString& subframeName = "") {
    TTree* outTree;
    return BuildCompositeIndex(mainTree, subframeTree, columns, subframeName, outTree);
}

// ============================================================================
// Schema Loading
// ============================================================================

/**
 * Load schema from embedded ADF_SCHEMA in ROOT file
 * 
 * @param file   Open TFile containing the schema
 * @param tree   TTree to apply aliases to
 * @return       True if schema found and loaded
 */
Bool_t LoadEmbeddedSchema(TFile* file, TTree* tree) {
    if (!file || !tree) return kFALSE;
    
    // Try to get embedded schema (written by Python save_schema_to_root)
    TObjString* schemaObj = (TObjString*)file->Get("ADF_SCHEMA");
    if (!schemaObj) {
        // Also check in tree's UserInfo (older format)
        TList* userInfo = tree->GetUserInfo();
        if (userInfo && userInfo->GetEntries() > 0) {
            TObjString* obj = dynamic_cast<TObjString*>(userInfo->At(0));
            if (obj) {
                TString json = obj->GetString();
                if (json.Contains("aliases") || json.Contains("columns")) {
                    return LoadSchemaFromJSON(tree, json);
                }
            }
        }
        return kFALSE;
    }
    
    TString json = schemaObj->GetString();
    return LoadSchemaFromJSON(tree, json);
}

/**
 * Parse JSON schema and apply to tree
 * 
 * Robust parser that handles both compact and formatted JSON
 */
Bool_t LoadSchemaFromJSON(TTree* tree, const TString& json) {
    SchemaInfo& schema = g_schemaRegistry[tree];
    schema.aliases.clear();
    schema.subframeIndices.clear();
    
    // Parse aliases section
    TString aliasSection = ExtractJSONObject(json, "aliases");
    if (aliasSection.Length() > 0) {
        // Find each alias definition
        Ssiz_t pos = 0;
        while (pos < aliasSection.Length()) {
            // Find next key
            Ssiz_t keyStart = aliasSection.Index("\"", pos);
            if (keyStart == kNPOS) break;
            
            Ssiz_t keyEnd = aliasSection.Index("\"", keyStart + 1);
            if (keyEnd == kNPOS) break;
            
            TString aliasName = aliasSection(keyStart + 1, keyEnd - keyStart - 1);
            
            // Skip internal keys
            if (aliasName.BeginsWith("__")) {
                pos = keyEnd + 1;
                continue;
            }
            
            // Find expression (look for "expr" in the value object)
            Ssiz_t objStart = aliasSection.Index("{", keyEnd);
            if (objStart == kNPOS) {
                pos = keyEnd + 1;
                continue;
            }
            
            // Find matching closing brace
            Int_t depth = 1;
            Ssiz_t objEnd = objStart + 1;
            while (depth > 0 && objEnd < aliasSection.Length()) {
                if (aliasSection[objEnd] == '{') depth++;
                if (aliasSection[objEnd] == '}') depth--;
                objEnd++;
            }
            
            TString valueObj = aliasSection(objStart, objEnd - objStart);
            TString expr = ExtractJSONString(valueObj, "expr");
            
            if (expr.Length() > 0) {
                schema.aliases[aliasName] = expr;
                // Apply alias to tree
                tree->SetAlias(aliasName.Data(), expr.Data());
            }
            
            pos = objEnd;
        }
    }
    
    // Also check for "columns" section (v2 schema format)
    TString columnsSection = ExtractJSONObject(json, "columns");
    if (columnsSection.Length() > 0) {
        Ssiz_t pos = 0;
        while (pos < columnsSection.Length()) {
            Ssiz_t keyStart = columnsSection.Index("\"", pos);
            if (keyStart == kNPOS) break;
            
            Ssiz_t keyEnd = columnsSection.Index("\"", keyStart + 1);
            if (keyEnd == kNPOS) break;
            
            TString colName = columnsSection(keyStart + 1, keyEnd - keyStart - 1);
            
            if (colName.BeginsWith("__")) {
                pos = keyEnd + 1;
                continue;
            }
            
            Ssiz_t objStart = columnsSection.Index("{", keyEnd);
            if (objStart == kNPOS) {
                pos = keyEnd + 1;
                continue;
            }
            
            Int_t depth = 1;
            Ssiz_t objEnd = objStart + 1;
            while (depth > 0 && objEnd < columnsSection.Length()) {
                if (columnsSection[objEnd] == '{') depth++;
                if (columnsSection[objEnd] == '}') depth--;
                objEnd++;
            }
            
            TString valueObj = columnsSection(objStart, objEnd - objStart);
            TString expr = ExtractJSONString(valueObj, "expr");
            
            if (expr.Length() > 0) {
                schema.aliases[colName] = expr;
                tree->SetAlias(colName.Data(), expr.Data());
            }
            
            pos = objEnd;
        }
    }
    
    // Parse subframes section
    TString subframesSection = ExtractJSONObject(json, "subframes");
    if (subframesSection.Length() > 0) {
        Ssiz_t pos = 0;
        while (pos < subframesSection.Length()) {
            Ssiz_t keyStart = subframesSection.Index("\"", pos);
            if (keyStart == kNPOS) break;
            
            Ssiz_t keyEnd = subframesSection.Index("\"", keyStart + 1);
            if (keyEnd == kNPOS) break;
            
            TString sfName = subframesSection(keyStart + 1, keyEnd - keyStart - 1);
            
            // Skip metadata
            if (sfName == "__meta__") {
                pos = keyEnd + 1;
                continue;
            }
            
            // Find subframe object
            Ssiz_t objStart = subframesSection.Index("{", keyEnd);
            if (objStart == kNPOS) {
                pos = keyEnd + 1;
                continue;
            }
            
            Int_t depth = 1;
            Ssiz_t objEnd = objStart + 1;
            while (depth > 0 && objEnd < subframesSection.Length()) {
                if (subframesSection[objEnd] == '{') depth++;
                if (subframesSection[objEnd] == '}') depth--;
                objEnd++;
            }
            
            TString sfObj = subframesSection(objStart, objEnd - objStart);
            
            // Extract index columns - try both "index" and "subframe_indices"
            std::vector<TString> indexCols = ExtractJSONArray(sfObj, "index");
            if (indexCols.empty()) {
                indexCols = ExtractJSONArray(sfObj, "subframe_indices");
            }
            
            if (!indexCols.empty()) {
                schema.subframeIndices[sfName] = indexCols;
            }
            
            pos = objEnd;
        }
    }
    
    schema.loaded = true;
    std::cout << "  Schema loaded: " << schema.aliases.size() << " aliases, " 
              << schema.subframeIndices.size() << " subframes" << std::endl;
    
    // Print subframe index info
    for (const auto& [name, cols] : schema.subframeIndices) {
        std::cout << "    Subframe '" << name << "' index: [";
        for (size_t i = 0; i < cols.size(); i++) {
            std::cout << cols[i];
            if (i < cols.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    
    return kTRUE;
}

/**
 * Load schema from external JSON file
 */
Bool_t LoadSchema(TTree* tree, const char* schemaPath) {
    if (!tree) {
        std::cerr << "Error: NULL tree" << std::endl;
        return kFALSE;
    }
    
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
    return LoadSchemaFromJSON(tree, json);
}

// ============================================================================
// Main Loading Functions
// ============================================================================

/**
 * Load an AliasDataFrame ROOT file with automatic schema and composite index support
 * 
 * Features:
 * - Automatically detects and loads embedded ADF_SCHEMA
 * - Builds N-key composite indices for subframes with >2 index columns
 * - Attaches subframes as friend trees
 * - For read-only files, clones subframes to memory for index building
 * 
 * @param filename  Path to ROOT file exported from AliasDataFrame
 * @param treename  Name of main tree (default: "tree")
 * @return          Pointer to main TTree with friends attached
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
    
    std::cout << "Loading ADF tree: " << filename << std::endl;
    
    // Report file access mode
    if (!f->IsWritable()) {
        std::cout << "  File opened read-only (composite indices will use in-memory clones)" << std::endl;
    }
    
    // Try to load embedded schema
    Bool_t hasSchema = LoadEmbeddedSchema(f, mainTree);
    SchemaInfo* schema = hasSchema ? &g_schemaRegistry[mainTree] : nullptr;
    
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
    
    // Attach each subframe as friend with appropriate indexing
    for (const auto& sfName : subframeNames) {
        TString sfTreeName = prefix + sfName;
        TTree* sfTree = (TTree*)f->Get(sfTreeName);
        
        if (!sfTree) continue;
        
        std::cout << "  Subframe '" << sfName << "':" << std::endl;
        
        // Determine index columns
        std::vector<TString> indexCols;
        
        // Priority 1: Use schema if available
        if (schema && schema->subframeIndices.count(sfName)) {
            indexCols = schema->subframeIndices[sfName];
            
            // Deduplicate index columns (handles Python schema bug where
            // single-key index stored as ["gid", "gid"] instead of ["gid"])
            {
                std::vector<TString> uniqueCols;
                for (const auto& col : indexCols) {
                    bool isDuplicate = false;
                    for (const auto& u : uniqueCols) {
                        if (u == col) { isDuplicate = true; break; }
                    }
                    if (!isDuplicate) uniqueCols.push_back(col);
                }
                if (uniqueCols.size() != indexCols.size()) {
                    std::cout << "    WARNING: Duplicate index columns removed: [";
                    for (size_t i = 0; i < indexCols.size(); i++) {
                        std::cout << indexCols[i];
                        if (i < indexCols.size() - 1) std::cout << ", ";
                    }
                    std::cout << "] → [";
                    for (size_t i = 0; i < uniqueCols.size(); i++) {
                        std::cout << uniqueCols[i];
                        if (i < uniqueCols.size() - 1) std::cout << ", ";
                    }
                    std::cout << "]" << std::endl;
                }
                indexCols = uniqueCols;
            }
            std::cout << "    Index from schema: [";
            for (size_t i = 0; i < indexCols.size(); i++) {
                std::cout << indexCols[i];
                if (i < indexCols.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        // Priority 2: Auto-detect common index columns
        else {
            for (const char* idxCol : {"track_index", "index", "key", "id"}) {
                if (sfTree->GetBranch(idxCol) && mainTree->GetBranch(idxCol)) {
                    indexCols.push_back(idxCol);
                    // Check for second key
                    for (const char* idx2Col : {"firstTFOrbit", "firstTForbit", "orbit", "tf"}) {
                        if (sfTree->GetBranch(idx2Col) && mainTree->GetBranch(idx2Col)) {
                            indexCols.push_back(idx2Col);
                            break;
                        }
                    }
                    break;
                }
            }
            if (!indexCols.empty()) {
                std::cout << "    Index auto-detected: [";
                for (size_t i = 0; i < indexCols.size(); i++) {
                    std::cout << indexCols[i];
                    if (i < indexCols.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            }
        }
        
        // Build index (composite if >2 columns)
        TTree* friendTree = sfTree;  // May be replaced with clone
        if (!indexCols.empty()) {
            BuildCompositeIndex(mainTree, sfTree, indexCols, sfName, friendTree);
        } else {
            std::cout << "    No index columns found (linear scan)" << std::endl;
        }
        
        // Add as friend (use clone if created)
        mainTree->AddFriend(friendTree, sfName.Data());
    }
    
    std::cout << "Loaded '" << treename << "' with " << subframeNames.size() 
              << " subframes" << std::endl;
    
    return mainTree;
}

/**
 * Load ADF tree with explicit index column specification
 * (Backward compatible - for manual index control)
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
        
        std::cout << "  Subframe '" << sfName << "':" << std::endl;
        TTree* friendTree = sfTree;
        BuildCompositeIndex(mainTree, sfTree, cols, sfName, friendTree);
        mainTree->AddFriend(friendTree, sfName.Data());
    }
    
    return mainTree;
}

// ============================================================================
// Introspection Functions
// ============================================================================

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
        // Skip internal composite key branches
        TString name = br->GetName();
        if (!name.BeginsWith("__")) {
            std::cout << "  " << br->GetName() << std::endl;
        }
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
                TString name = br->GetName();
                if (!name.BeginsWith("__")) {
                    std::cout << "  " << fe->GetName() << "." << br->GetName() << std::endl;
                }
            }
        }
    }
}

/**
 * Describe loaded schema
 */
void DescribeSchema(TTree* tree) {
    if (!tree) {
        std::cout << "No tree provided" << std::endl;
        return;
    }
    
    auto it = g_schemaRegistry.find(tree);
    if (it == g_schemaRegistry.end() || !it->second.loaded) {
        std::cout << "No schema loaded for this tree" << std::endl;
        return;
    }
    
    SchemaInfo& schema = it->second;
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "Schema for tree: " << tree->GetName() << std::endl;
    std::cout << "======================================" << std::endl;
    
    std::cout << "\nAliases (" << schema.aliases.size() << "):" << std::endl;
    for (const auto& [name, expr] : schema.aliases) {
        std::cout << "  " << name << " = " << expr << std::endl;
    }
    
    std::cout << "\nSubframe indices (" << schema.subframeIndices.size() << "):" << std::endl;
    for (const auto& [name, cols] : schema.subframeIndices) {
        std::cout << "  " << name << ": [";
        for (size_t i = 0; i < cols.size(); i++) {
            std::cout << cols[i];
            if (i < cols.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (cols.size() > 2) {
            std::cout << " (composite index)";
        }
        std::cout << std::endl;
    }
    
    std::cout << "======================================\n" << std::endl;
}

/**
 * Describe data (entries, memory usage)
 */
void DescribeData(TTree* tree, const char* sortBy = "name") {
    if (!tree) return;
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "Data summary: " << tree->GetName() << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << "Entries: " << tree->GetEntries() << std::endl;
    
    // Print friend info
    TList* friends = tree->GetListOfFriends();
    if (friends && friends->GetEntries() > 0) {
        std::cout << "\nFriend trees:" << std::endl;
        TIter next(friends);
        TFriendElement* fe;
        while ((fe = (TFriendElement*)next())) {
            TTree* ft = fe->GetTree();
            printf("  %-20s %10lld entries\n", fe->GetName(), ft->GetEntries());
        }
    }
    
    // Report in-memory clones
    if (!g_memFiles.empty()) {
        std::cout << "\nIn-memory subframe clones: " << g_memFiles.size() << std::endl;
        Long64_t totalMem = 0;
        for (auto* mf : g_memFiles) {
            totalMem += mf->GetSize();
        }
        std::cout << "  Total memory: " << totalMem / 1024.0 / 1024.0 << " MB" << std::endl;
    }
    
    std::cout << "======================================\n" << std::endl;
}

/**
 * Clean up in-memory files
 */
void CleanupMemFiles() {
    std::cout << "Cleaning up " << g_memFiles.size() << " in-memory files..." << std::endl;
    for (auto* mf : g_memFiles) {
        delete mf;
    }
    g_memFiles.clear();
}

/**
 * Example usage function
 */
void ExampleUsage() {
    std::cout << R"(
=== AliasDataFrameTree.C Usage Examples (Phase 3.1) ===

1. Basic loading (automatic schema + composite index):
   TTree* tree = LoadADFTree("calibration.root", "tree");
   tree->Draw("dy:dz");  // Aliases work, N-key indices work

2. Load with external schema:
   TTree* tree = LoadADFTree("data.root", "tree");
   LoadSchema(tree, "custom_schema.json");

3. Describe schema and data:
   DescribeSchema(tree);
   DescribeData(tree);

4. Print all branches:
   PrintADFBranches(tree);

5. Manual index specification (override schema):
   std::map<TString, std::vector<TString>> idx;
   idx["DITS0FitSide"] = {"row", "drift25", "side", "firstTFOrbit"};
   TTree* tree = LoadADFTreeWithIndex("data.root", "tree", idx);

6. Draw with subframe columns:
   tree->Draw("mX - DITS0FitSide.mX");
   tree->Draw("dy:dz", "DITS0FitSide.valid == 1");

7. Cleanup in-memory clones:
   CleanupMemFiles();

Features (Phase 3.1):
- Automatic embedded schema loading
- N-key composite indices (cardinality-based packing)
- In-memory cloning for read-only files
- Handles sparse indices (e.g., firstTFOrbit) correctly
- Merge-safe (hadd) composite keys

)" << std::endl;
}

// Auto-run when loaded
#ifndef __CINT__
void AliasDataFrameTree() {
    std::cout << "AliasDataFrameTree.C loaded (Phase 3.1: In-memory Index)." << std::endl;
    std::cout << "Features: N-key composite index, in-memory cloning for read-only files" << std::endl;
    std::cout << "Run ExampleUsage() for help." << std::endl;
}
#endif
