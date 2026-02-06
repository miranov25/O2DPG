/**
 * @file O2ResidualHelpersLinkDef.h
 * @brief ROOT dictionary linkdef for O2 residual types
 */

#ifdef __CLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// Enable RVec instantiations for O2 types
#pragma link C++ class ROOT::VecOps::RVec<o2::tpc::DetInfoResid>+;
#pragma link C++ class ROOT::VecOps::RVec<o2::tpc::UnbinnedResid>+;

// Enable our helper namespace and functions
#pragma link C++ namespace RDataFrameDSL;
#pragma link C++ namespace RDataFrameDSL::O2ResidualHelpers;
#pragma link C++ function RDataFrameDSL::O2ResidualHelpers::ExtractQMaxTPC;
#pragma link C++ function RDataFrameDSL::O2ResidualHelpers::ExtractQTotTPC;

#endif
