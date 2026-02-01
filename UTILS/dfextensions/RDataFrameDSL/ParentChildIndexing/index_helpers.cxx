/**
 * @file index_helpers.cxx
 * @brief Explicit template instantiations for index_helpers
 * 
 * This file provides explicit instantiations for common types
 * to be compiled into libIndexHelpers.so
 */

#include "index_helpers.h"

namespace RDataFrameDSL {
namespace IndexHelpers {

// ============================================================================
// Explicit instantiations for ExpandToChildren
// ============================================================================

template RVec<float> ExpandToChildren<float>(
    const RVec<float>&, const RVec<int>&, const RVec<int>&);

template RVec<double> ExpandToChildren<double>(
    const RVec<double>&, const RVec<int>&, const RVec<int>&);

template RVec<int> ExpandToChildren<int>(
    const RVec<int>&, const RVec<int>&, const RVec<int>&);

template RVec<short> ExpandToChildren<short>(
    const RVec<short>&, const RVec<int>&, const RVec<int>&);

template RVec<unsigned char> ExpandToChildren<unsigned char>(
    const RVec<unsigned char>&, const RVec<int>&, const RVec<int>&);

// ============================================================================
// Explicit instantiations for GatherByIndex
// ============================================================================

template RVec<float> GatherByIndex<float>(
    const RVec<float>&, const RVec<int>&, float);

template RVec<double> GatherByIndex<double>(
    const RVec<double>&, const RVec<int>&, double);

template RVec<int> GatherByIndex<int>(
    const RVec<int>&, const RVec<int>&, int);

template RVec<short> GatherByIndex<short>(
    const RVec<short>&, const RVec<int>&, short);

// ============================================================================
// Explicit instantiations for GetRange
// ============================================================================

template RVec<float> GetRange<float>(const RVec<float>&, int, int);

template RVec<double> GetRange<double>(const RVec<double>&, int, int);

template RVec<int> GetRange<int>(const RVec<int>&, int, int);

}  // namespace IndexHelpers
}  // namespace RDataFrameDSL
