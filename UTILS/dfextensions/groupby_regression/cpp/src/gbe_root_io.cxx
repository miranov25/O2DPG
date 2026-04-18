// cpp/src/gbe_root_io.cxx
//
// Phase 13.18.GB Layer B implementation.
//
// Reads a dfGB TTree from a .root file via TFile + TTreeReader, builds
// a Layer A GroupByRegressionEvaluator from the rows, and registers it
// in a process-static map keyed by user-chosen model name.
//
// FP-4 (TTreeReader -> TBranch fallback) NOT invoked in Turn 6
// initial implementation. If TTreeReader binding fails on alma2,
// switch to direct TBranch::GetEntry per the proposal v1.1 §6.3.

#include "gbe_root_io.h"
#include "gbe_kernel.hpp"
#include "json_reader.hpp"

#include <TBranch.h>
#include <TFile.h>
#include <TInterpreter.h>
#include <TKey.h>
#include <TObjString.h>
#include <TString.h>
#include <TTree.h>

#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace GBE {

namespace {

// Process-static registry. Function-local static avoids static-init
// order issues. Mutex protects the map; individual evaluator queries
// are read-only after registration so no per-eval locking required.
struct Registry {
    std::map<std::string, std::unique_ptr<gbe::GroupByRegressionEvaluator>> models;
    std::mutex mu;
};

Registry& registry() {
    static Registry R;
    return R;
}

// Parse the architect-blessed sidecar name pattern.
std::string sidecar_name_for(const std::string& tree_name) {
    return tree_name + "__gbreg_schema";
}

gbe::MethodMode parse_method(const std::string& s) {
    if (s == "lookup") return gbe::MethodMode::Lookup;
    if (s == "linear") return gbe::MethodMode::Linear;
    throw std::invalid_argument("unknown method: " + s);
}

gbe::BoundsMode parse_bounds(const std::string& s) {
    if (s == "nan") return gbe::BoundsMode::Nan;
    if (s == "clamp") return gbe::BoundsMode::Clamp;
    throw std::invalid_argument("unknown bounds: " + s);
}

// Build the set of expected coefficient column names for a given schema.
std::set<std::string> required_coeff_names(
    const std::vector<std::string>& targets,
    const std::vector<std::string>& predictor_columns,
    const std::string& suffix,
    bool fit_intercept)
{
    std::set<std::string> names;
    for (const auto& t : targets) {
        if (fit_intercept) {
            names.insert(t + "_intercept" + suffix);
        }
        for (const auto& p : predictor_columns) {
            names.insert(t + "_slope_" + p + suffix);
        }
    }
    return names;
}

// Walk the TTree and produce the SubframeRow vector consumed by the
// Layer A constructor.
//
// Branch handling:
//   - group_columns: read as int64_t (treated as Long64_t in TTree);
// FP-4 INVOKED — TTreeReader → TBranch::SetBranchAddress fallback.
//
// TTreeReaderValue<Long64_t> failed on alma2 ROOT 6.36 with uproot-
// written int64 branches (TTreeReader status 7 / type-mismatch).
// Using SetBranchAddress with type-matched buffers:
//   - group columns: Long64_t* (uproot writes np.int64 as Long64_t)
//   - coefficient columns: Double_t* (uproot writes np.float64 as Double_t)
//
// Returns rows by value; on error throws std::runtime_error.
std::vector<gbe::SubframeRow> read_subframe_rows(
    TTree* tree,
    const std::vector<std::string>& group_columns,
    const std::set<std::string>& required_coeffs)
{
    if (!tree) {
        throw std::runtime_error("read_subframe_rows: null TTree");
    }

    const Long64_t nentries = tree->GetEntries();

    // Group columns: Long64_t buffers (matches uproot's np.int64 encoding)
    std::vector<Long64_t> gc_bufs(group_columns.size(), 0);
    for (std::size_t d = 0; d < group_columns.size(); ++d) {
        TBranch* br = tree->GetBranch(group_columns[d].c_str());
        if (!br) {
            throw std::runtime_error(
                "missing group column branch: " + group_columns[d]);
        }
        tree->SetBranchAddress(group_columns[d].c_str(), &gc_bufs[d]);
    }

    // Coefficient columns: Double_t buffers (matches np.float64)
    std::vector<std::string> cc_names_ordered(required_coeffs.begin(),
                                               required_coeffs.end());
    std::vector<Double_t> cc_bufs(cc_names_ordered.size(), 0.0);
    for (std::size_t i = 0; i < cc_names_ordered.size(); ++i) {
        TBranch* br = tree->GetBranch(cc_names_ordered[i].c_str());
        if (!br) {
            throw std::runtime_error(
                "missing coefficient column branch: " + cc_names_ordered[i]);
        }
        tree->SetBranchAddress(cc_names_ordered[i].c_str(), &cc_bufs[i]);
    }

    std::vector<gbe::SubframeRow> rows;
    rows.reserve(static_cast<std::size_t>(nentries));
    for (Long64_t entry = 0; entry < nentries; ++entry) {
        tree->GetEntry(entry);
        gbe::SubframeRow row;
        for (std::size_t d = 0; d < group_columns.size(); ++d) {
            row.group_values[group_columns[d]] =
                static_cast<int64_t>(gc_bufs[d]);
        }
        for (std::size_t i = 0; i < cc_names_ordered.size(); ++i) {
            row.coeff_values[cc_names_ordered[i]] =
                static_cast<double>(cc_bufs[i]);
        }
        rows.push_back(std::move(row));
    }

    return rows;
}

// Open a TFile (defers to TFile::Open so URLs work).
// Returns nullptr on failure (TFile::Open prints its own diagnostic).
std::unique_ptr<TFile> open_file(const std::string& path) {
    std::unique_ptr<TFile> f(TFile::Open(path.c_str(), "READ"));
    if (!f || f->IsZombie()) {
        return nullptr;
    }
    return f;
}

// Pull the TTree out of an open file. Returns nullptr if missing or
// not a tree.
TTree* fetch_tree(TFile& f, const std::string& tree_name) {
    TObject* obj = f.Get(tree_name.c_str());
    if (!obj) return nullptr;
    return dynamic_cast<TTree*>(obj);
}

// Pull the sidecar TObjString and return its decoded JSON content.
// Returns empty string on missing-sidecar (caller must check).
std::string fetch_sidecar(TFile& f, const std::string& tree_name) {
    const std::string sidecar = sidecar_name_for(tree_name);
    TObject* obj = f.Get(sidecar.c_str());
    if (!obj) return std::string();
    auto* tos = dynamic_cast<TObjString*>(obj);
    if (!tos) return std::string();
    return std::string(tos->GetString().Data());
}

// Common load path for both Option A and Option B.
bool load_into_registry(
    const std::string& model_name,
    TTree* tree,
    gbe::ModelSchema schema,
    gbe::MethodMode method,
    gbe::BoundsMode bounds)
{
    auto coeffs = required_coeff_names(
        schema.targets, schema.predictor_columns,
        schema.suffix, schema.fit_intercept);

    std::vector<gbe::SubframeRow> rows;
    try {
        rows = read_subframe_rows(tree, schema.group_columns, coeffs);
    } catch (const std::exception& e) {
        std::cerr << "GBE::load_model: TTree read error for model '"
                  << model_name << "': " << e.what() << std::endl;
        return false;
    }

    std::unique_ptr<gbe::GroupByRegressionEvaluator> ev;
    try {
        ev = std::make_unique<gbe::GroupByRegressionEvaluator>(
            std::move(schema), std::move(rows), method, bounds);
    } catch (const std::exception& e) {
        std::cerr << "GBE::load_model: Layer A construction failed for '"
                  << model_name << "': " << e.what() << std::endl;
        return false;
    }

    auto& R = registry();
    std::lock_guard<std::mutex> lk(R.mu);
    R.models[model_name] = std::move(ev);
    return true;
}

}  // anonymous namespace

// -------------- public API --------------

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
    const std::string& bounds)
{
    auto file = open_file(path);
    if (!file) {
        std::cerr << "GBE::load_model_explicit: cannot open '" << path
                  << "'" << std::endl;
        return false;
    }
    TTree* tree = fetch_tree(*file, tree_name);
    if (!tree) {
        std::cerr << "GBE::load_model_explicit: tree '" << tree_name
                  << "' not found in '" << path << "'" << std::endl;
        return false;
    }

    gbe::ModelSchema schema;
    schema.group_columns = group_columns;
    schema.predictor_columns = predictor_columns;
    schema.targets = targets;
    schema.suffix = suffix;
    schema.fit_intercept = fit_intercept;

    gbe::MethodMode m;
    gbe::BoundsMode b;
    try {
        m = parse_method(method);
        b = parse_bounds(bounds);
    } catch (const std::exception& e) {
        std::cerr << "GBE::load_model_explicit: " << e.what() << std::endl;
        return false;
    }

    return load_into_registry(model_name, tree, std::move(schema), m, b)
           && declare_eval_stub(model_name);
}

bool load_model_from_metadata(
    const std::string& model_name,
    const std::string& path,
    const std::string& tree_name,
    const std::string& method,
    const std::string& bounds)
{
    auto file = open_file(path);
    if (!file) {
        std::cerr << "GBE::load_model_from_metadata: cannot open '"
                  << path << "'" << std::endl;
        return false;
    }
    TTree* tree = fetch_tree(*file, tree_name);
    if (!tree) {
        std::cerr << "GBE::load_model_from_metadata: tree '" << tree_name
                  << "' not found in '" << path << "'" << std::endl;
        return false;
    }

    const std::string sidecar_json = fetch_sidecar(*file, tree_name);
    if (sidecar_json.empty()) {
        std::cerr << "GBE::load_model_from_metadata: sidecar '"
                  << sidecar_name_for(tree_name)
                  << "' not found in '" << path << "'" << std::endl;
        return false;
    }

    gbe::ModelSchema schema;
    try {
        const gbe::JsonValue root = gbe::parse_json(sidecar_json);
        for (const auto& v : root.at("group_columns").as_array()) {
            schema.group_columns.push_back(v.as_string());
        }
        for (const auto& v : root.at("predictor_columns").as_array()) {
            schema.predictor_columns.push_back(v.as_string());
        }
        for (const auto& v : root.at("targets").as_array()) {
            schema.targets.push_back(v.as_string());
        }
        schema.suffix = root.at("suffix").as_string();
        schema.fit_intercept = root.at("fit_intercept").as_bool();
    } catch (const std::exception& e) {
        std::cerr << "GBE::load_model_from_metadata: schema decode error: "
                  << e.what() << std::endl;
        return false;
    }

    gbe::MethodMode m;
    gbe::BoundsMode b;
    try {
        m = parse_method(method);
        b = parse_bounds(bounds);
    } catch (const std::exception& e) {
        std::cerr << "GBE::load_model_from_metadata: " << e.what()
                  << std::endl;
        return false;
    }

    return load_into_registry(model_name, tree, std::move(schema), m, b)
           && declare_eval_stub(model_name);
}

bool has_model(const std::string& model_name) {
    auto& R = registry();
    std::lock_guard<std::mutex> lk(R.mu);
    return R.models.find(model_name) != R.models.end();
}

const gbe::GroupByRegressionEvaluator* get_model(
    const std::string& model_name)
{
    auto& R = registry();
    std::lock_guard<std::mutex> lk(R.mu);
    auto it = R.models.find(model_name);
    if (it == R.models.end()) return nullptr;
    return it->second.get();
}

std::vector<std::string> list_models() {
    std::vector<std::string> out;
    auto& R = registry();
    std::lock_guard<std::mutex> lk(R.mu);
    for (const auto& kv : R.models) out.push_back(kv.first);
    return out;
}

bool unload_model(const std::string& model_name) {
    auto& R = registry();
    std::lock_guard<std::mutex> lk(R.mu);
    return R.models.erase(model_name) > 0;
}

void clear_models() {
    auto& R = registry();
    std::lock_guard<std::mutex> lk(R.mu);
    R.models.clear();
}

// -------------- Turn 7: gInterpreter stub + eval_on_tree --------------

// Helper: convert natural-label to fractional compact index.
// Handles lookup vs linear and nan vs clamp bounds modes.
double map_natural_to_compact(const std::string& model_name,
                               int dim, double natural_pos)
{
    const auto* ev = get_model(model_name);
    if (!ev) return std::nan("");

    const auto& bc = ev->bin_centers();
    const std::size_t udim = static_cast<std::size_t>(dim);
    if (udim >= bc.size()) return std::nan("");

    const auto& centers = bc[udim];
    const std::size_t N = centers.size();
    if (N == 0) return std::nan("");

    if (ev->method() == gbe::MethodMode::Lookup) {
        // Exact integer match via remap
        int64_t int_pos = static_cast<int64_t>(std::llround(natural_pos));
        const auto& rm = ev->remap()[udim];
        auto it = rm.find(int_pos);
        if (it != rm.end()) {
            return static_cast<double>(it->second);
        }
        // Not found — handle bounds
        if (ev->bounds() == gbe::BoundsMode::Clamp) {
            if (natural_pos <= static_cast<double>(centers.front()))
                return 0.0;
            return static_cast<double>(N - 1);
        }
        return std::nan("");
    }

    // Linear mode: find fractional position in bin_centers space
    double first = static_cast<double>(centers.front());
    double last  = static_cast<double>(centers.back());

    if (natural_pos < first) {
        if (ev->bounds() == gbe::BoundsMode::Clamp) return 0.0;
        return std::nan("");
    }
    if (natural_pos > last) {
        if (ev->bounds() == gbe::BoundsMode::Clamp)
            return static_cast<double>(N - 1);
        return std::nan("");
    }

    // Interval search: find i such that centers[i] <= pos <= centers[i+1]
    for (std::size_t i = 0; i + 1 < N; ++i) {
        double lo = static_cast<double>(centers[i]);
        double hi = static_cast<double>(centers[i + 1]);
        if (natural_pos >= lo && natural_pos <= hi) {
            double frac = (hi == lo) ? 0.0
                                     : (natural_pos - lo) / (hi - lo);
            return static_cast<double>(i) + frac;
        }
    }
    // Exact match with last bin
    return static_cast<double>(N - 1);
}

bool declare_eval_stub(const std::string& model_name) {
    const auto* ev = get_model(model_name);
    if (!ev) {
        std::cerr << "GBE::declare_eval_stub: model '" << model_name
                  << "' not found in registry" << std::endl;
        return false;
    }

    const auto& schema = ev->schema();
    const std::size_t n_gc = schema.group_columns.size();
    const std::size_t n_pred = schema.predictor_columns.size();
    const std::size_t arity = n_gc + n_pred;

    // Build the function source string.
    // The stub calls map_natural_to_compact for each group dimension,
    // which handles lookup/linear × nan/clamp correctly (including
    // fractional positions for linear interpolation).

    std::ostringstream src;
    src << "namespace GBE {\n";
    src << "double eval_" << model_name << "(";
    for (std::size_t i = 0; i < arity; ++i) {
        if (i > 0) src << ", ";
        src << "double a" << i;
    }
    src << ") {\n";
    src << "  const auto* ev = GBE::get_model(\"" << model_name << "\");\n";
    src << "  if (!ev) return std::nan(\"\");\n";

    // Map each group column from natural label to compact index
    for (std::size_t d = 0; d < n_gc; ++d) {
        src << "  double c" << d << " = GBE::map_natural_to_compact(\""
            << model_name << "\", " << d << ", a" << d << ");\n";
        src << "  if (std::isnan(c" << d << ")) return std::nan(\"\");\n";
    }

    // Predictor values passed through directly
    src << "  std::vector<double> pv = {";
    for (std::size_t i = 0; i < n_pred; ++i) {
        if (i > 0) src << ", ";
        src << "a" << (n_gc + i);
    }
    src << "};\n";

    if (ev->method() == gbe::MethodMode::Lookup) {
        src << "  std::vector<int64_t> pos = {";
        for (std::size_t d = 0; d < n_gc; ++d) {
            if (d > 0) src << ", ";
            src << "static_cast<int64_t>(c" << d << ")";
        }
        src << "};\n";
        src << "  auto r = ev->evaluate_lookup(pos, pv);\n";
    } else {
        src << "  std::vector<double> pos = {";
        for (std::size_t d = 0; d < n_gc; ++d) {
            if (d > 0) src << ", ";
            src << "c" << d;
        }
        src << "};\n";
        src << "  auto r = ev->evaluate_linear(pos, pv);\n";
    }
    src << "  return r.empty() ? std::nan(\"\") : r[0];\n";
    src << "}\n";
    src << "} // namespace GBE\n";

    const std::string code = src.str();

    if (!gInterpreter->Declare(code.c_str())) {
        std::cerr << "GBE::declare_eval_stub: gInterpreter->Declare failed "
                  << "for model '" << model_name << "'. Generated code:\n"
                  << code << std::endl;
        return false;
    }

    return true;
}

std::vector<double> eval_on_tree(
    const std::string& model_name,
    TTree* tree,
    const std::vector<std::string>& column_names)
{
    const auto* ev = get_model(model_name);
    if (!ev) {
        std::cerr << "GBE::eval_on_tree: model '" << model_name
                  << "' not found" << std::endl;
        return {};
    }
    if (!tree) {
        std::cerr << "GBE::eval_on_tree: null TTree" << std::endl;
        return {};
    }

    const auto& schema = ev->schema();
    const std::size_t n_gc = schema.group_columns.size();
    const std::size_t n_pred = schema.predictor_columns.size();
    const std::size_t expected_cols = n_gc + n_pred;

    if (column_names.size() != expected_cols) {
        std::cerr << "GBE::eval_on_tree: column_names size "
                  << column_names.size() << " != expected "
                  << expected_cols << " (group=" << n_gc
                  << " + pred=" << n_pred << ")" << std::endl;
        return {};
    }

    const Long64_t nentries = tree->GetEntries();

    // Group columns as Long64_t, predictor columns as Double_t
    std::vector<Long64_t> gc_bufs(n_gc, 0);
    std::vector<Double_t> pv_bufs(n_pred, 0.0);

    for (std::size_t d = 0; d < n_gc; ++d) {
        TBranch* br = tree->GetBranch(column_names[d].c_str());
        if (!br) {
            std::cerr << "GBE::eval_on_tree: branch '"
                      << column_names[d] << "' not found" << std::endl;
            return {};
        }
        tree->SetBranchAddress(column_names[d].c_str(), &gc_bufs[d]);
    }
    for (std::size_t i = 0; i < n_pred; ++i) {
        TBranch* br = tree->GetBranch(column_names[n_gc + i].c_str());
        if (!br) {
            std::cerr << "GBE::eval_on_tree: branch '"
                      << column_names[n_gc + i] << "' not found" << std::endl;
            return {};
        }
        tree->SetBranchAddress(column_names[n_gc + i].c_str(), &pv_bufs[i]);
    }

    const double kNaN = std::nan("");

    std::vector<double> result;
    result.reserve(static_cast<std::size_t>(nentries));

    for (Long64_t entry = 0; entry < nentries; ++entry) {
        tree->GetEntry(entry);

        // Map each group column via map_natural_to_compact
        bool out_of_grid = false;

        if (ev->method() == gbe::MethodMode::Lookup) {
            std::vector<int64_t> pos(n_gc);
            for (std::size_t d = 0; d < n_gc; ++d) {
                double compact = map_natural_to_compact(
                    model_name, static_cast<int>(d),
                    static_cast<double>(gc_bufs[d]));
                if (std::isnan(compact)) {
                    out_of_grid = true; break;
                }
                pos[d] = static_cast<int64_t>(compact);
            }
            if (out_of_grid) {
                result.push_back(kNaN);
                continue;
            }
            std::vector<double> pv(pv_bufs.begin(), pv_bufs.end());
            auto r = ev->evaluate_lookup(pos, pv);
            result.push_back(r.empty() ? kNaN : r[0]);
        } else {
            std::vector<double> pos(n_gc);
            for (std::size_t d = 0; d < n_gc; ++d) {
                double compact = map_natural_to_compact(
                    model_name, static_cast<int>(d),
                    static_cast<double>(gc_bufs[d]));
                if (std::isnan(compact)) {
                    out_of_grid = true; break;
                }
                pos[d] = compact;
            }
            if (out_of_grid) {
                result.push_back(kNaN);
                continue;
            }
            std::vector<double> pv(pv_bufs.begin(), pv_bufs.end());
            auto r = ev->evaluate_linear(pos, pv);
            result.push_back(r.empty() ? kNaN : r[0]);
        }
    }

    return result;
}

}  // namespace GBE
