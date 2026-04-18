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
// user-chosen string name. Turn 7 hooks gInterpreter->Declare per-model
// stubs and adds eval_on_tree tabular entry point.
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

class TTree;  // ROOT forward decl at global scope (NOT inside namespace GBE)

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

// -- Turn 7: per-model gInterpreter stub + tabular entry point --

// Generates and JIT-compiles a free function
//   double GBE::eval_<model_name>(double a0, double a1, ...)
// via gInterpreter->Declare. Arity = group_columns.size() +
// predictor_columns.size(). Called automatically by load_model_*.
// Returns true if the stub compiled successfully.
//
// The generated stub looks up the model in the static registry,
// remaps the double args to compact indices (for group columns) or
// passes them as predictor values, and calls evaluate_lookup or
// evaluate_linear. Returns the first target's prediction (or NaN).
//
// FP-5: if gInterpreter->Declare fails for arity >= 6, fall back to
// a vector-arg wrapper and document.
bool declare_eval_stub(const std::string& model_name);

// Evaluate a loaded model on every entry of an input TTree. Returns
// a vector<double> of length tree->GetEntries() containing the first
// target's prediction per entry.
//
// column_names: ordered list of branch names in the input tree
//   corresponding to [group_columns..., predictor_columns...] in that
//   order. Must have exactly group_columns.size() +
//   predictor_columns.size() elements.
//
// Returns empty vector on error (diagnostic printed to stderr).
std::vector<double> eval_on_tree(
    const std::string& model_name,
    TTree* tree,
    const std::vector<std::string>& column_names);

// Convert a natural-label position (e.g. sector=15) to its fractional
// compact index for the given model dimension. Handles bounds mode:
//   - Lookup + NaN:   exact match required; not-found → NaN
//   - Lookup + Clamp: exact match or clamp to nearest edge
//   - Linear + NaN:   interpolated position in [0, N-1]; out-of-range → NaN
//   - Linear + Clamp: out-of-range clamped to [0, N-1]
// Used by the gInterpreter stubs and eval_on_tree.
double map_natural_to_compact(const std::string& model_name,
                               int dim, double natural_pos);

}  // namespace GBE

#endif  // GBE_ROOT_IO_H
