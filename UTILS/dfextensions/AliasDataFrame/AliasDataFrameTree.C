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
 * Phase 3: Multi-key composite index support using cardinality-based packing
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
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
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
 * NOTE: Only builds index on subframe tree, NOT on main tree.
 * ROOT's friend mechanism will use the subframe's index for lookups.
 * 
 * @param mainTree      Main tree (used to create matching key column)
 * @param subframeTree  Subframe TTree to index
 * @param columns       Index column names from schema
 * @param subframeName  Name of subframe (for unique branch naming)
 * @return              true if successful, false otherwise
 */
Bool_t BuildCompositeIndex(TTree* mainTree, TTree* subframeTree, 
                           const std::vector<TString>& columns,
                           const TString& subframeName = "") {
    if (columns.size() <= 2) {
        // Use native ROOT BuildIndex for 1-2 keys
        if (columns.size() == 1) {
            subframeTree->BuildIndex(columns[0].Data());
            std::cout << "    BuildIndex(" << columns[0] << ")" << std::endl;
        } else if (columns.size() == 2) {
            subframeTree->BuildIndex(columns[0].Data(), columns[1].Data());
            std::cout << "    BuildIndex(" << columns[0] << ", " << columns[1] << ")" << std::endl;
        }
        return kTRUE;
    }
    
    // N > 2 keys: use cardinality-based composite index
    // IMPORTANT: This requires adding branches to trees, which only works
    // if the trees are writable (opened with "UPDATE" or created in memory)
    
    Long64_t nEntries = subframeTree->GetEntries();
    size_t nCols = columns.size();
    
    std::cout << "    BuildCompositeIndex: " << nCols << " keys, " 
              << nEntries << " entries" << std::endl;
    
    // Check if trees are writable (must be opened with "UPDATE" or "RECREATE")
    TFile* sfFile = subframeTree->GetCurrentFile();
    Bool_t isWritable = kFALSE;
    if (sfFile) {
        TString option = sfFile->GetOption();
        option.ToUpper();
        isWritable = option.Contains("UPDATE") || option.Contains("RECREATE") || 
                     option.Contains("CREATE") || option.Contains("NEW");
    }
    
    if (!isWritable) {
        std::cout << "    NOTE: File opened read-only. Cannot build " << nCols 
                  << "-key composite index." << std::endl;
        std::cout << "    Subframe '" << subframeName << "' will use linear scan (slower)." << std::endl;
        std::cout << "    For indexed access, use: TFile::Open(filename, \"UPDATE\")" << std::endl;
        return kFALSE;
    }
    
    // Verify all columns exist and are integer types
    std::vector<TLeaf*> leaves(nCols);
    for (size_t i = 0; i < nCols; i++) {
        TLeaf* leaf = subframeTree->GetLeaf(columns[i].Data());
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
    
    // =========================================================
    // PASS 1: Build value → code dictionaries for each column
    // =========================================================
    std::vector<std::map<Long64_t, Long64_t>> valueToCodes(nCols);
    
    for (Long64_t entry = 0; entry < nEntries; entry++) {
        subframeTree->GetEntry(entry);
        for (size_t i = 0; i < nCols; i++) {
            Long64_t value = (Long64_t)leaves[i]->GetValue();
            auto& dict = valueToCodes[i];
            if (dict.find(value) == dict.end()) {
                dict[value] = dict.size();  // Assign next code: 0, 1, 2, ...
            }
        }
    }
    
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
    // PASS 2: Create composite key branch in SUBFRAME ONLY
    // =========================================================
    // Use unique branch name per subframe to avoid collisions
    TString keyBranchName = TString::Format("__adf_key_%s__", 
        subframeName.Length() > 0 ? subframeName.Data() : "idx");
    
    Long64_t compositeKey;
    TBranch* keyBranch = subframeTree->Branch(keyBranchName.Data(), &compositeKey, 
                                              TString::Format("%s/L", keyBranchName.Data()).Data());
    
    if (!keyBranch) {
        std::cerr << "ERROR: Failed to create composite key branch on subframe tree" << std::endl;
        return kFALSE;
    }
    
    for (Long64_t entry = 0; entry < nEntries; entry++) {
        subframeTree->GetEntry(entry);
        
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
    
    // Build index on subframe's composite key
    subframeTree->BuildIndex(keyBranchName.Data());
    
    // =========================================================
    // Create matching composite key column in MAIN TREE
    // (Required for ROOT friend tree join to work)
    // =========================================================
    // Check for columns in main tree - they might be branches OR aliases
    std::vector<TLeaf*> mainLeaves(nCols);
    Bool_t allColumnsInMain = kTRUE;
    for (size_t i = 0; i < nCols; i++) {
        mainLeaves[i] = mainTree->GetLeaf(columns[i].Data());
        if (!mainLeaves[i]) {
            // Check if it's an alias
            TList* aliasList = mainTree->GetListOfAliases();
            Bool_t isAlias = aliasList && aliasList->FindObject(columns[i].Data());
            if (isAlias) {
                std::cout << "      Note: '" << columns[i] << "' is an alias in main tree" << std::endl;
            } else {
                std::cerr << "WARNING: Column '" << columns[i] 
                          << "' not found in main tree (neither branch nor alias)." << std::endl;
            }
            allColumnsInMain = kFALSE;
        }
    }
    
    if (allColumnsInMain) {
        Long64_t mainCompositeKey;
        TBranch* mainKeyBranch = mainTree->Branch(keyBranchName.Data(), &mainCompositeKey,
                                                   TString::Format("%s/L", keyBranchName.Data()).Data());
        
        if (!mainKeyBranch) {
            std::cerr << "WARNING: Failed to create composite key branch on main tree" << std::endl;
        } else {
            Long64_t mainEntries = mainTree->GetEntries();
            for (Long64_t entry = 0; entry < mainEntries; entry++) {
                mainTree->GetEntry(entry);
                
                mainCompositeKey = 0;
                Long64_t multiplier = 1;
                for (size_t i = 0; i < nCols; i++) {
                    Long64_t value = (Long64_t)mainLeaves[i]->GetValue();
                    // Map value to code (use -1 for unknown values to ensure no match)
                    auto it = valueToCodes[i].find(value);
                    Long64_t code = (it != valueToCodes[i].end()) ? it->second : -1;
                    if (code == -1) {
                        mainCompositeKey = -1;  // No match possible
                        break;
                    }
                    mainCompositeKey += code * multiplier;
                    multiplier *= cardinalities[i];
                }
                
                mainKeyBranch->Fill();
            }
        }
        
        // NOTE: Do NOT call BuildIndex on main tree!
        // ROOT will use the subframe's index when evaluating friend expressions
    }
    
    std::cout << "    Composite index built successfully" << std::endl;
    return kTRUE;
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
 * Robust parser that handles multiple schema formats:
 * - v1 legacy: "subframe_indices": {"T": "col" or ["col1", "col2"]}
 * - v1 nested: "subframes": {"T": {"index": [...]}}
 * - v2: "subframes": {"T": {"index": [...], "columns": {...}}}
 * 
 * Also handles "aliases" (v1) vs "columns" with "expr" (v2)
 */
Bool_t LoadSchemaFromJSON(TTree* tree, const TString& json) {
    SchemaInfo& schema = g_schemaRegistry[tree];
    schema.aliases.clear();
    schema.subframeIndices.clear();
    
    // ========================================================================
    // PRIORITY 1: Parse "subframe_indices" (v1 legacy format)
    // Format: "subframe_indices": {"T": "col", "R": ["col1", "col2"], ...}
    // ========================================================================
    TString subframeIndicesSection = ExtractJSONObject(json, "subframe_indices");
    if (subframeIndicesSection.Length() > 0) {
        Ssiz_t pos = 0;
        while (pos < subframeIndicesSection.Length()) {
            // Find opening quote of subframe name
            Ssiz_t nameStart = subframeIndicesSection.Index("\"", pos);
            if (nameStart == kNPOS) break;
            
            Ssiz_t nameEnd = subframeIndicesSection.Index("\"", nameStart + 1);
            if (nameEnd == kNPOS) break;
            
            TString sfName = subframeIndicesSection(nameStart + 1, nameEnd - nameStart - 1);
            
            // Find colon after name
            Ssiz_t colonPos = subframeIndicesSection.Index(":", nameEnd);
            if (colonPos == kNPOS) {
                pos = nameEnd + 1;
                continue;
            }
            
            // Look for either "[" (array) or "\"" (single string) after colon
            Ssiz_t arrayStart = subframeIndicesSection.Index("[", colonPos);
            Ssiz_t stringStart = subframeIndicesSection.Index("\"", colonPos + 1);
            
            std::vector<TString> indices;
            
            // Determine if it's an array or single string
            if (arrayStart != kNPOS && (stringStart == kNPOS || arrayStart < stringStart)) {
                // It's an array: ["col1", "col2", ...]
                Ssiz_t arrayEnd = subframeIndicesSection.Index("]", arrayStart);
                if (arrayEnd != kNPOS) {
                    TString arrayContent = subframeIndicesSection(arrayStart + 1, arrayEnd - arrayStart - 1);
                    TObjArray* tokens = arrayContent.Tokenize(",");
                    for (Int_t i = 0; i < tokens->GetEntries(); i++) {
                        TString token = ((TObjString*)tokens->At(i))->GetString();
                        token.ReplaceAll("\"", "");
                        token.ReplaceAll(" ", "");
                        token.ReplaceAll("\n", "");
                        token.ReplaceAll("\t", "");
                        if (token.Length() > 0) {
                            indices.push_back(token);
                        }
                    }
                    delete tokens;
                    pos = arrayEnd + 1;
                } else {
                    pos = nameEnd + 1;
                }
            } else if (stringStart != kNPOS) {
                // It's a single string: "col"
                Ssiz_t stringEnd = subframeIndicesSection.Index("\"", stringStart + 1);
                if (stringEnd != kNPOS) {
                    TString singleIndex = subframeIndicesSection(stringStart + 1, stringEnd - stringStart - 1);
                    if (singleIndex.Length() > 0) {
                        indices.push_back(singleIndex);
                    }
                    pos = stringEnd + 1;
                } else {
                    pos = nameEnd + 1;
                }
            } else {
                pos = nameEnd + 1;
                continue;
            }
            
            if (!indices.empty()) {
                schema.subframeIndices[sfName] = indices;
            }
        }
    }
    
    // ========================================================================
    // PRIORITY 2: Parse "subframes" with nested "index" (v1 nested / v2 format)
    // Only if subframe_indices didn't provide the info
    // Format: "subframes": { "NAME": { "index": ["col1", "col2", ...] }, ... }
    // ========================================================================
    if (schema.subframeIndices.empty()) {
        TString subframesSection = ExtractJSONObject(json, "subframes");
        
        if (subframesSection.Length() > 0 && subframesSection.Contains("\"index\"")) {
            Ssiz_t pos = 0;
            while (pos < subframesSection.Length()) {
                Ssiz_t nameStart = subframesSection.Index("\"", pos);
                if (nameStart == kNPOS) break;
                
                Ssiz_t nameEnd = subframesSection.Index("\"", nameStart + 1);
                if (nameEnd == kNPOS) break;
                
                TString sfName = subframesSection(nameStart + 1, nameEnd - nameStart - 1);
                
                // Skip JSON field keys
                if (sfName == "index" || sfName == "tree_name" || sfName == "dtype" || 
                    sfName == "expr" || sfName == "columns" || sfName.Length() == 0) {
                    pos = nameEnd + 1;
                    continue;
                }
                
                Ssiz_t objStart = subframesSection.Index("{", nameEnd);
                if (objStart == kNPOS) break;
                
                // Find matching closing brace
                Int_t depth = 1;
                Ssiz_t objEnd = objStart + 1;
                while (objEnd < subframesSection.Length() && depth > 0) {
                    if (subframesSection[objEnd] == '{') depth++;
                    if (subframesSection[objEnd] == '}') depth--;
                    objEnd++;
                }
                
                TString sfObject = subframesSection(objStart, objEnd - objStart);
                std::vector<TString> indices = ExtractJSONArray(sfObject, "index");
                if (indices.empty()) {
                    TString singleIndex = ExtractJSONString(sfObject, "index");
                    if (singleIndex.Length() > 0) {
                        indices.push_back(singleIndex);
                    }
                }
                
                if (!indices.empty()) {
                    schema.subframeIndices[sfName] = indices;
                }
                
                pos = objEnd;
            }
        }
    }
    
    // ========================================================================
    // Parse aliases from "aliases" section (v1 format)
    // Format: "aliases": {"name": "expr", ...}
    // ========================================================================
    TString aliasesSection = ExtractJSONObject(json, "aliases");
    if (aliasesSection.Length() > 0) {
        Ssiz_t pos = 0;
        while (pos < aliasesSection.Length()) {
            Ssiz_t nameStart = aliasesSection.Index("\"", pos);
            if (nameStart == kNPOS) break;
            
            Ssiz_t nameEnd = aliasesSection.Index("\"", nameStart + 1);
            if (nameEnd == kNPOS) break;
            
            TString aliasName = aliasesSection(nameStart + 1, nameEnd - nameStart - 1);
            
            Ssiz_t colonPos = aliasesSection.Index(":", nameEnd);
            if (colonPos == kNPOS) {
                pos = nameEnd + 1;
                continue;
            }
            
            Ssiz_t exprStart = aliasesSection.Index("\"", colonPos);
            if (exprStart == kNPOS) {
                pos = nameEnd + 1;
                continue;
            }
            
            // Find end of expression (handle escaped quotes)
            Ssiz_t exprEnd = exprStart + 1;
            while (exprEnd < aliasesSection.Length()) {
                if (aliasesSection[exprEnd] == '\"' && aliasesSection[exprEnd - 1] != '\\') {
                    break;
                }
                exprEnd++;
            }
            
            if (exprEnd < aliasesSection.Length()) {
                TString expr = aliasesSection(exprStart + 1, exprEnd - exprStart - 1);
                if (expr.Length() > 0) {
                    schema.aliases[aliasName] = expr;
                    tree->SetAlias(aliasName.Data(), expr.Data());
                }
            }
            
            pos = exprEnd + 1;
        }
    }
    
    // ========================================================================
    // Parse aliases from "columns" section (v2 format) - only if aliases empty
    // Format: "columns": {"name": {"expr": "...", "dtype": "..."}, ...}
    // ========================================================================
    if (schema.aliases.empty()) {
        TString columnsSection = ExtractJSONObject(json, "columns");
        if (columnsSection.Length() > 0) {
            Ssiz_t pos = 0;
            while (pos < columnsSection.Length()) {
                Ssiz_t nameStart = columnsSection.Index("\"", pos);
                if (nameStart == kNPOS) break;
                
                Ssiz_t nameEnd = columnsSection.Index("\"", nameStart + 1);
                if (nameEnd == kNPOS) break;
                
                TString colName = columnsSection(nameStart + 1, nameEnd - nameStart - 1);
                
                if (colName == "dtype" || colName == "expr" || colName == "constant" || 
                    colName == "index" || colName.Length() == 0) {
                    pos = nameEnd + 1;
                    continue;
                }
                
                Ssiz_t objStart = columnsSection.Index("{", nameEnd);
                if (objStart == kNPOS) break;
                
                TString between = columnsSection(nameEnd + 1, objStart - nameEnd - 1);
                between.ReplaceAll(" ", "");
                between.ReplaceAll("\n", "");
                between.ReplaceAll("\t", "");
                if (!between.BeginsWith(":")) {
                    pos = nameEnd + 1;
                    continue;
                }
                
                Int_t depth = 1;
                Ssiz_t objEnd = objStart + 1;
                while (objEnd < columnsSection.Length() && depth > 0) {
                    if (columnsSection[objEnd] == '{') depth++;
                    if (columnsSection[objEnd] == '}') depth--;
                    objEnd++;
                }
                
                TString colObject = columnsSection(objStart, objEnd - objStart);
                TString expr = ExtractJSONString(colObject, "expr");
                if (expr.Length() > 0 && expr != "null") {
                    schema.aliases[colName] = expr;
                    tree->SetAlias(colName.Data(), expr.Data());
                }
                
                pos = objEnd;
            }
        }
    }
    
    schema.loaded = true;
    
    std::cout << "  Schema loaded: " << schema.aliases.size() << " aliases, "
              << schema.subframeIndices.size() << " subframes" << std::endl;
    
    // Debug: print what we found
    for (const auto& [name, indices] : schema.subframeIndices) {
        std::cout << "    Subframe '" << name << "' index: [";
        for (size_t i = 0; i < indices.size(); i++) {
            std::cout << indices[i];
            if (i < indices.size() - 1) std::cout << ", ";
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
        if (!indexCols.empty()) {
            BuildCompositeIndex(mainTree, sfTree, indexCols, sfName);
        } else {
            std::cout << "    No index columns found (linear scan)" << std::endl;
        }
        
        // Add as friend
        mainTree->AddFriend(sfTree, sfName.Data());
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
        BuildCompositeIndex(mainTree, sfTree, cols, sfName);
        mainTree->AddFriend(sfTree, sfName.Data());
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
    
    std::cout << "======================================\n" << std::endl;
}

/**
 * Example usage function
 */
void ExampleUsage() {
    std::cout << R"(
=== AliasDataFrameTree.C Usage Examples (Phase 3) ===

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

Features (Phase 3):
- Automatic embedded schema loading
- N-key composite indices (cardinality-based packing)
- Handles sparse indices (e.g., firstTFOrbit) correctly
- Merge-safe (hadd) composite keys

)" << std::endl;
}

// Auto-run when loaded
#ifndef __CINT__
void AliasDataFrameTree() {
    std::cout << "AliasDataFrameTree.C loaded (Phase 3: Composite Index)." << std::endl;
    std::cout << "Features: N-key composite index, automatic schema loading" << std::endl;
    std::cout << "Run ExampleUsage() for help." << std::endl;
}
#endif
