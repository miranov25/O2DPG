// cpp/demo/demo_tabular.C
//
// Phase 13.18.GB Demo 3: Tabular evaluation via eval_on_tree.
//
// Demonstrates batch evaluation of a loaded model over every entry
// in an input TTree, returning a std::vector<double> of predictions.
//
// Usage:
//   root -l -b -q 'demo/demo_tabular.C("path/to/model.root", "path/to/input.root")'
//
// If no input.root provided, uses the model's own dfGB tree as a
// trivial self-prediction test (evaluates at the training grid points).

#include <iostream>
#include <vector>
#include <string>
#include <cmath>

void demo_tabular(const char* model_path = "",
                  const char* input_path = "") {
    if (std::string(model_path).empty()) {
        std::cout << "Usage: root -l -b -q "
                  << "'demo/demo_tabular.C(\"model.root\", \"input.root\")'"
                  << std::endl;
        std::cout << "Run demo_pyroot.py first to generate model.root"
                  << std::endl;
        return;
    }

    // Load library + model
    gSystem->Load("./libGroupByRegressionEvaluator.so");
    bool ok = GBE::load_model_from_metadata("tab_demo", model_path,
                                             "dfGB", "lookup", "nan");
    if (!ok) {
        std::cerr << "ERROR: load failed" << std::endl;
        return;
    }

    // Determine which file to read entries from
    std::string ipath = (std::string(input_path).empty())
                        ? std::string(model_path)
                        : std::string(input_path);
    std::string tree_name = (std::string(input_path).empty())
                            ? "dfGB" : "input";

    TFile* f = TFile::Open(ipath.c_str(), "READ");
    TTree* tree = (TTree*)f->Get(tree_name.c_str());
    if (!tree) {
        std::cerr << "ERROR: tree '" << tree_name << "' not found in "
                  << ipath << std::endl;
        return;
    }

    // Build column_names from the model schema
    auto* ev = GBE::get_model("tab_demo");
    std::vector<std::string> cols;
    for (auto& gc : ev->schema().group_columns) cols.push_back(gc);
    for (auto& pc : ev->schema().predictor_columns) cols.push_back(pc);

    std::cout << "Evaluating " << tree->GetEntries() << " entries..."
              << std::endl;
    std::cout << "  columns: ";
    for (auto& c : cols) std::cout << c << " ";
    std::cout << std::endl;

    // Check that all required branches exist in the input tree
    bool all_found = true;
    for (auto& c : cols) {
        if (!tree->GetBranch(c.c_str())) {
            std::cerr << "  WARNING: branch '" << c << "' not found in "
                      << "input tree. eval_on_tree will fail for this column."
                      << std::endl;
            all_found = false;
        }
    }

    if (!all_found && std::string(input_path).empty()) {
        // Self-prediction mode: dfGB tree has group columns but not
        // bare predictor columns. Print what we can.
        std::cout << "\n  Note: self-prediction mode. The dfGB tree has "
                  << "group columns but not bare predictor columns."
                  << std::endl;
        std::cout << "  For a real tabular demo, provide an input.root "
                  << "with all required branches." << std::endl;
        // Still attempt — eval_on_tree will return empty on missing branch
    }

    auto preds = GBE::eval_on_tree("tab_demo", tree, cols);

    std::cout << "\n--- Predictions (first 10) ---" << std::endl;
    int n = std::min((int)preds.size(), 10);
    int nan_count = 0;
    for (int i = 0; i < n; ++i) {
        std::cout << "  entry " << i << ": " << preds[i];
        if (std::isnan(preds[i])) {
            std::cout << " (NaN)";
            nan_count++;
        }
        std::cout << std::endl;
    }
    std::cout << "\nTotal: " << preds.size() << " predictions, "
              << nan_count << " NaN in first " << n << std::endl;

    GBE::clear_models();
    f->Close();
    std::cout << "\n=== Demo 3 (tabular): DONE ===" << std::endl;
}
