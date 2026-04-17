// cpp/LinkDef.h
//
// Phase 13.18.GB Layer B — ROOT dictionary directives for the GBE
// namespace public API. Generated dictionary lets PyROOT and ROOT
// macros call GBE::load_model_explicit(...), GBE::load_model_from_metadata(...),
// GBE::has_model(...), etc., and dispatches them to the compiled C++.
//
// Turn 7 will add directives for the per-model gInterpreter stubs and
// for GBE::eval_on_tree once those land.

#ifdef __ROOTCLING__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

// Free functions in namespace GBE
#pragma link C++ namespace GBE+;

#pragma link C++ function GBE::load_model_explicit;
#pragma link C++ function GBE::load_model_from_metadata;
#pragma link C++ function GBE::has_model;
#pragma link C++ function GBE::get_model;
#pragma link C++ function GBE::list_models;
#pragma link C++ function GBE::unload_model;
#pragma link C++ function GBE::clear_models;

// Layer A class — opaque to PyROOT in Turn 6 (returned as
// const GroupByRegressionEvaluator* from get_model). Turn 7 will
// expose evaluate_lookup / evaluate_linear via gInterpreter stubs.
#pragma link C++ namespace gbe+;
#pragma link C++ class gbe::GroupByRegressionEvaluator+;

#endif  // __ROOTCLING__
