// cpp/include/gbe_root_io.h
//
// Phase 13.18.GB Layer B — ROOT-dependent IO for the GroupByRegression
// evaluator. Provides:
//
//   - load_model_explicit(...)        Option B: caller supplies schema
//   - load_model_from_metadata(...)   Option A: schema discovered from
//                                     a TObjString sidecar named
//                                     <tree_name>__gbreg_schema
//
//   - get_model(name)                 fetch a previously-loaded model
//   - has_model(name) / list_models() registry introspection
//   - unload_model(name) / clear_models()  registry teardown
//
// All loaded models live in a process-static registry, addressable by
// user-chosen string name. This is what Turn 7 will hook the
// `gInterpreter->Declare` per-model stub into.
//
// Layer B INCLUDES ROOT headers — DO NOT include this header from
// Layer A sources or from anything that compiles to WASM. The
// wasm_lint Makefile target ignores this file by design.
//
// Design choice notes (Q1/Q2/Q3 from Turn 6 kickoff):
//   Q1: TFile path resolution defers to TFile::Open, so URLs like
//       root:// or alien:// are supported transparently.
//   Q2: schema sidecar JSON is parsed via the Layer A json_reader
//       (Layer A is statically linked into Layer B — no duplication).
//   Q3: process-static registry; ownership is library-managed; PyROOT
//       holds a string name, not a pointer.

#ifndef GBE_ROOT_IO_H
#define GBE_ROOT_IO_H

#include <string>
#include <vector>

namespace gbe {
class GroupByRegressionEvaluator;
}

namespace GBE {

// Method / bounds enums match Layer A but are duplicated here as plain
// strings at the Layer B API to keep the public PyROOT surface trivial
// to call. ("lookup" / "linear", "nan" / "clamp")

// -- Option B: explicit schema --
//
// Loads a dfGB model from a ROOT file and registers it under the given
// model name. Caller supplies the full schema. Returns true on success.
//
// path        : TFile path or URL (resolved via TFile::Open)
// tree_name   : name of the TTree inside the file
// group_columns / predictor_columns / targets / suffix / fit_intercept :
//   schema as in Layer A's ModelSchema
// method      : "lookup" or "linear"
// bounds      : "nan" or "clamp"
//
// On error: prints a diagnostic to stderr and returns false. (Throwing
// across the PyROOT boundary is supported but produces less friendly
// error messages; a bool return + printed diagnostic is the usual
// ROOT-glue idiom.)
bool load_model_explicit(
    const std::string& model_name,
    const std::string& path,
    const std::string& tree_name,
    const std::vector<std::string>& group_columns,
    const std::vector<std::string>& predictor_columns,
    const std::vector<std::string>& targets,
    const std::string& suffix,
    bool fit_intercept,
    const std::string& method,
    const std::string& bounds);

// -- Option A: schema discovered from sidecar --
//
// Reads a TObjString sidecar named "<tree_name>__gbreg_schema" from the
// same TFile, decodes the JSON inside, then loads the model with the
// recovered schema.
//
// method / bounds are still passed at load time because they are not
// part of the dfGB model itself — they describe how to query it.
bool load_model_from_metadata(
    const std::string& model_name,
    const std::string& path,
    const std::string& tree_name,
    const std::string& method,
    const std::string& bounds);

// -- Registry access --

bool has_model(const std::string& model_name);

// Returns nullptr if the model is not registered. Caller must NOT
// delete the returned pointer; ownership stays with the registry.
const gbe::GroupByRegressionEvaluator* get_model(
    const std::string& model_name);

std::vector<std::string> list_models();

bool unload_model(const std::string& model_name);
void clear_models();

}  // namespace GBE

#endif  // GBE_ROOT_IO_H
