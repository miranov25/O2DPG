/**
 * @file o2_residual_helpers.h
 * @brief O2-specific helpers for TPC residual analysis
 * 
 * This library provides RVec dictionary and helper functions for O2 TPC 
 * residual data structures.
 * 
 * Prerequisites:
 *   - O2 environment loaded (alienv enter O2Physics/latest)
 *   - MS_GSL_ROOT, O2_ROOT environment variables set
 * 
 * Usage:
 *   ROOT.gSystem.Load("libO2ResidualHelpers.so")
 *   ROOT.gInterpreter.Declare('#include "o2_residual_helpers.h"')
 */

#ifndef O2_RESIDUAL_HELPERS_H
#define O2_RESIDUAL_HELPERS_H

#include <ROOT/RVec.hxx>
#include "SpacePoints/TrackInterpolation.h"

namespace RDataFrameDSL {
namespace O2ResidualHelpers {

/**
 * Extract qMaxTPC from DetInfoResid vector using O2 interface.
 * 
 * @param detInfo Vector of DetInfoResid (one per residual)
 * @return Vector of qMaxTPC values
 */
inline ROOT::RVec<int> ExtractQMaxTPC(const ROOT::RVec<o2::tpc::DetInfoResid>& detInfo) {
    ROOT::RVec<int> result(detInfo.size());
    for (size_t i = 0; i < detInfo.size(); ++i) {
        result[i] = detInfo[i].qMaxTPC();
    }
    return result;
}

/**
 * Extract qTotTPC from DetInfoResid vector using O2 interface.
 * 
 * @param detInfo Vector of DetInfoResid (one per residual)
 * @return Vector of qTotTPC values
 */
inline ROOT::RVec<int> ExtractQTotTPC(const ROOT::RVec<o2::tpc::DetInfoResid>& detInfo) {
    ROOT::RVec<int> result(detInfo.size());
    for (size_t i = 0; i < detInfo.size(); ++i) {
        result[i] = detInfo[i].qTotTPC();
    }
    return result;
}

}  // namespace O2ResidualHelpers
}  // namespace RDataFrameDSL

#endif  // O2_RESIDUAL_HELPERS_H
