/**
 * @file index_helpers.h
 * @brief Parent-Child Index Helpers for RDataFrameDSL
 * 
 * Phase 13.7.B: Provides C++ helpers for navigating hierarchical data.
 * 
 * Patterns supported:
 *   1. Offset+Count (residuals): ExpandParentIndex, ExpandToChildren
 *   2. Direct Index (MC): GatherByIndex
 *   3. Range (MC daughters): GetDaughterRange
 *   4. Reverse lookup: CountChildren, CountChildrenByCategory
 * 
 * Usage:
 *   #include "index_helpers.h"
 *   auto parentIdx = RDataFrameDSL::IndexHelpers::ExpandParentIndex(firstIdx, nEntries);
 * 
 * @author RDataFrameDSL Team
 * @date 2026-02-01
 */

#ifndef RDATAFRAMEDSL_INDEX_HELPERS_H
#define RDATAFRAMEDSL_INDEX_HELPERS_H

#include <ROOT/RVec.hxx>

namespace RDataFrameDSL {
namespace IndexHelpers {

using namespace ROOT::VecOps;

// ============================================================================
// Pattern 1a: Offset-only (compute nEntries from consecutive offsets)
// Use when nEntries is buggy/unavailable, only offsets are reliable
// ============================================================================

/**
 * Compute nEntries from consecutive offset values.
 * 
 * @param offsets     First index for each parent [nParents]
 * @param totalSize   Total number of children (e.g., res.dy.size())
 * @return Number of children per parent [nParents]
 * 
 * Example:
 *   offsets   = [0, 158, 309, 460]
 *   totalSize = 600
 *   Output    = [158, 151, 151, 140]  (last = 600 - 460)
 */
inline RVec<int> ComputeEntriesFromOffsets(const RVec<unsigned int>& offsets, int totalSize) {
    if (offsets.empty()) return RVec<int>{};
    
    RVec<int> nEntries(offsets.size());
    for (size_t i = 0; i < offsets.size() - 1; ++i) {
        nEntries[i] = static_cast<int>(offsets[i+1]) - static_cast<int>(offsets[i]);
    }
    // Last element: goes to totalSize
    nEntries[offsets.size() - 1] = totalSize - static_cast<int>(offsets[offsets.size() - 1]);
    return nEntries;
}

// Overload for signed int offsets
inline RVec<int> ComputeEntriesFromOffsets(const RVec<int>& offsets, int totalSize) {
    if (offsets.empty()) return RVec<int>{};
    
    RVec<int> nEntries(offsets.size());
    for (size_t i = 0; i < offsets.size() - 1; ++i) {
        nEntries[i] = offsets[i+1] - offsets[i];
    }
    // Last element: goes to totalSize
    nEntries[offsets.size() - 1] = totalSize - offsets[offsets.size() - 1];
    return nEntries;
}

/**
 * Build parent index from offsets only (without explicit nEntries).
 * Combines ComputeEntriesFromOffsets + ExpandParentIndex.
 * 
 * @param offsets     First index for each parent [nParents]
 * @param totalSize   Total number of children
 * @return Parent index for each child [totalSize]
 */
inline RVec<int> ExpandParentIndexFromOffsets(const RVec<unsigned int>& offsets, int totalSize) {
    auto nEntries = ComputeEntriesFromOffsets(offsets, totalSize);
    
    RVec<int> parentIdx(totalSize);
    int childPos = 0;
    for (size_t parent = 0; parent < offsets.size(); ++parent) {
        for (int i = 0; i < nEntries[parent]; ++i) {
            parentIdx[childPos++] = static_cast<int>(parent);
        }
    }
    return parentIdx;
}

// Overload for signed int offsets
inline RVec<int> ExpandParentIndexFromOffsets(const RVec<int>& offsets, int totalSize) {
    auto nEntries = ComputeEntriesFromOffsets(offsets, totalSize);
    
    RVec<int> parentIdx(totalSize);
    int childPos = 0;
    for (size_t parent = 0; parent < offsets.size(); ++parent) {
        for (int i = 0; i < nEntries[parent]; ++i) {
            parentIdx[childPos++] = static_cast<int>(parent);
        }
    }
    return parentIdx;
}

/**
 * Expand parent values to child level using offsets only.
 * Combines ComputeEntriesFromOffsets + ExpandToChildren.
 * 
 * @param parentValues  Values per parent [nParents]
 * @param offsets       First index for each parent [nParents]
 * @param totalSize     Total number of children
 * @return Values expanded to child level [totalSize]
 */
template<typename T>
RVec<T> ExpandToChildrenFromOffsets(const RVec<T>& parentValues,
                                     const RVec<unsigned int>& offsets,
                                     int totalSize) {
    auto nEntries = ComputeEntriesFromOffsets(offsets, totalSize);
    
    RVec<T> childValues(totalSize);
    int childPos = 0;
    for (size_t parent = 0; parent < parentValues.size(); ++parent) {
        for (int i = 0; i < nEntries[parent]; ++i) {
            childValues[childPos++] = parentValues[parent];
        }
    }
    return childValues;
}

// Overload for signed int offsets
template<typename T>
RVec<T> ExpandToChildrenFromOffsets(const RVec<T>& parentValues,
                                     const RVec<int>& offsets,
                                     int totalSize) {
    auto nEntries = ComputeEntriesFromOffsets(offsets, totalSize);
    
    RVec<T> childValues(totalSize);
    int childPos = 0;
    for (size_t parent = 0; parent < parentValues.size(); ++parent) {
        for (int i = 0; i < nEntries[parent]; ++i) {
            childValues[childPos++] = parentValues[parent];
        }
    }
    return childValues;
}

// ============================================================================
// Pattern 1: Offset + Count (Residuals, general parent-child)
// INVARIANT: Indices are contiguous (firstIdx[i+1] == firstIdx[i] + nEntries[i])
// ============================================================================

/**
 * Build child-to-parent index mapping.
 * 
 * @param firstIdx  First child index for each parent [nParents]
 * @param nEntries  Number of children for each parent [nParents]
 * @return Parent index for each child [sum(nEntries)]
 * 
 * Example:
 *   firstIdx  = [0, 3, 5]
 *   nEntries  = [3, 2, 4]
 *   Output    = [0,0,0, 1,1, 2,2,2,2]
 */
inline RVec<int> ExpandParentIndex(const RVec<int>& firstIdx, 
                                    const RVec<int>& nEntries) {
    int totalChildren = 0;
    for (auto n : nEntries) totalChildren += n;
    
    RVec<int> parentIdx(totalChildren);
    int childPos = 0;
    for (size_t parent = 0; parent < firstIdx.size(); ++parent) {
        for (int i = 0; i < nEntries[parent]; ++i) {
            parentIdx[childPos++] = static_cast<int>(parent);
        }
    }
    return parentIdx;
}

/**
 * Expand parent values to child level.
 * 
 * @param parentValues  Values per parent [nParents]
 * @param firstIdx      First child index for each parent [nParents]
 * @param nEntries      Number of children for each parent [nParents]
 * @return Values expanded to child level [sum(nEntries)]
 */
template<typename T>
RVec<T> ExpandToChildren(const RVec<T>& parentValues,
                          const RVec<int>& firstIdx,
                          const RVec<int>& nEntries) {
    int totalChildren = 0;
    for (auto n : nEntries) totalChildren += n;
    
    RVec<T> childValues(totalChildren);
    int childPos = 0;
    for (size_t parent = 0; parent < parentValues.size(); ++parent) {
        for (int i = 0; i < nEntries[parent]; ++i) {
            childValues[childPos++] = parentValues[parent];
        }
    }
    return childValues;
}

// ============================================================================
// Pattern 2: Direct Index Lookup (MC mother, TrackRef→MCTrack)
// Handles -1 (invalid) indices with fallback value
// ============================================================================

/**
 * Gather values by index with bounds checking.
 * 
 * @param values    Source values [nSources]
 * @param indices   Indices to gather [nResults], -1 = use fallback
 * @param fallback  Value to use for invalid indices
 * @return Gathered values [nResults]
 */
template<typename T>
RVec<T> GatherByIndex(const RVec<T>& values,
                       const RVec<int>& indices,
                       T fallback = T{}) {
    RVec<T> result(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        result[i] = (idx >= 0 && idx < static_cast<int>(values.size())) 
                    ? values[idx] 
                    : fallback;
    }
    return result;
}

// ============================================================================
// Pattern 3: Range [first, last] inclusive (MC daughters)
// ============================================================================

/**
 * Get elements in range [first, last] inclusive.
 * 
 * @param allValues     Source array
 * @param firstIdx      Start index (-1 = empty result)
 * @param lastIdx       End index inclusive (-1 = empty result)
 * @return Elements in range, or empty if invalid
 * 
 * Example: first=2, last=4 → returns elements [2,3,4]
 */
template<typename T>
RVec<T> GetRange(const RVec<T>& allValues,
                  int firstIdx,
                  int lastIdx) {
    if (firstIdx < 0 || lastIdx < 0) {
        return RVec<T>{};
    }
    if (firstIdx >= static_cast<int>(allValues.size()) || 
        lastIdx >= static_cast<int>(allValues.size())) {
        return RVec<T>{};
    }
    if (firstIdx > lastIdx) {
        return RVec<T>{};
    }
    return RVec<T>(allValues.begin() + firstIdx,
                   allValues.begin() + lastIdx + 1);
}

// ============================================================================
// Pattern 4: Reverse Lookup (MCTrack → TrackRefs)
// ============================================================================

/**
 * Count children per parent.
 * 
 * @param childToParent  Parent index for each child [nChildren]
 * @param nParents       Number of parents
 * @return Count of children per parent [nParents]
 */
inline RVec<int> CountChildren(const RVec<int>& childToParent, int nParents) {
    RVec<int> counts(nParents, 0);
    for (int parent : childToParent) {
        if (parent >= 0 && parent < nParents) {
            counts[parent]++;
        }
    }
    return counts;
}

/**
 * Count children by category (e.g., TrackRefs by detector).
 * 
 * @param childToParent   Parent index for each child [nChildren]
 * @param childCategory   Category for each child [nChildren]
 * @param nParents        Number of parents
 * @param targetCategory  Category to count
 * @return Count of matching children per parent [nParents]
 */
inline RVec<int> CountChildrenByCategory(const RVec<int>& childToParent,
                                          const RVec<int>& childCategory,
                                          int nParents,
                                          int targetCategory) {
    RVec<int> counts(nParents, 0);
    for (size_t i = 0; i < childToParent.size(); ++i) {
        int parent = childToParent[i];
        if (parent >= 0 && parent < nParents && childCategory[i] == targetCategory) {
            counts[parent]++;
        }
    }
    return counts;
}

}  // namespace IndexHelpers
}  // namespace RDataFrameDSL

#endif  // RDATAFRAMEDSL_INDEX_HELPERS_H
