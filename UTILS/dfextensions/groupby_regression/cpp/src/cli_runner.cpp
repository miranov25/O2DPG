// cpp/src/cli_runner.cpp — Turn 3 command-line subprocess binary.
//
// Invoked by cpp/tests/test_layer_a_lookup.py as a subprocess. Reads a
// fixture JSON from the path given on the command line, constructs a
// GroupByRegressionEvaluator, evaluates every query position, and
// writes a single structured JSON object to stdout containing the
// predictions plus diagnostic intermediates.
//
// Structured stdout JSON format:
//   {
//     "status": "ok" | "error",
//     "error":  "<message if status==error>",
//     "predictions": { "<target>": [v0, v1, ...], ... },
//     "actual_bin_centers": [[...], ...],
//     "actual_valid_mask_flat": [true, false, ...],
//     "actual_grid_shape": [N0, N1, ...]
//   }
//
// NaN values in predictions are serialized as the JSON string "NaN"
// (mirrors the fixture input format; FIXTURE_SPEC v1.0 §4).

#include "gbe_kernel.hpp"
#include "json_reader.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using gbe::JsonValue;

namespace {

// Serialize a double, using "NaN" string for NaN. Returns a string like
// '3.14159' or '"NaN"'. Uses high-precision %.17g format so the
// round-trip through the test harness is lossless.
std::string format_double(double v) {
    if (std::isnan(v)) return "\"NaN\"";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string format_int_array(const std::vector<int64_t>& v) {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << v[i];
    }
    oss << ']';
    return oss.str();
}

std::string format_int_array(const std::vector<int>& v) {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << v[i];
    }
    oss << ']';
    return oss.str();
}

std::string format_bool_array(const std::vector<bool>& v) {
    std::ostringstream oss;
    oss << '[';
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) oss << ',';
        oss << (v[i] ? "true" : "false");
    }
    oss << ']';
    return oss.str();
}

// Extract int64 value from JsonValue (supports JSON number only —
// fixture schema stores group indices as numbers, not strings).
int64_t as_int64(const JsonValue& v) {
    const double d = v.as_number();
    const int64_t r = static_cast<int64_t>(d);
    if (static_cast<double>(r) != d) {
        throw std::invalid_argument("expected integer-valued number");
    }
    return r;
}

// Convert a fixture "input" block into ModelSchema + rows + query positions.
struct FixtureBundle {
    gbe::ModelSchema schema;
    std::vector<gbe::SubframeRow> rows;
    std::vector<std::vector<int64_t>> query_positions;      // per-query compact idx
    std::vector<std::vector<double>>  predictor_values_per_query;
    gbe::MethodMode method;
    gbe::BoundsMode bounds;
};

FixtureBundle unpack_fixture(const JsonValue& root) {
    // Validate strict 5-key top level.
    const auto& top = root.as_object();
    static const char* required_top[] = {
        "fixture_id", "axis_values", "input",
        "expected_output", "intermediates", "metadata"
    };
    for (const char* k : required_top) {
        if (top.find(k) == top.end()) {
            throw std::invalid_argument(
                std::string("fixture missing top-level key: ") + k);
        }
    }
    // Reject unknown top-level keys (restricted-parser contract).
    for (const auto& kv : top) {
        bool ok = false;
        for (const char* k : required_top) {
            if (kv.first == k) { ok = true; break; }
        }
        if (!ok) {
            throw std::invalid_argument(
                "fixture has unknown top-level key: " + kv.first);
        }
    }

    const JsonValue& input = root.at("input");
    FixtureBundle b;

    // schema fields
    for (const auto& v : input.at("gb_columns").as_array()) {
        b.schema.group_columns.push_back(v.as_string());
    }
    for (const auto& v : input.at("predictor_columns").as_array()) {
        b.schema.predictor_columns.push_back(v.as_string());
    }
    for (const auto& v : input.at("targets").as_array()) {
        b.schema.targets.push_back(v.as_string());
    }
    b.schema.suffix = input.at("suffix").as_string();
    b.schema.fit_intercept = input.at("fit_intercept").as_bool();

    // method / bounds
    const std::string& m = input.at("method").as_string();
    if      (m == "lookup") b.method = gbe::MethodMode::Lookup;
    else if (m == "linear") b.method = gbe::MethodMode::Linear;
    else throw std::invalid_argument("unknown method: " + m);

    const std::string& bnd = input.at("bounds").as_string();
    if      (bnd == "nan")   b.bounds = gbe::BoundsMode::Nan;
    else if (bnd == "clamp") b.bounds = gbe::BoundsMode::Clamp;
    else throw std::invalid_argument("unknown bounds: " + bnd);

    // subframe_rows
    for (const auto& r : input.at("subframe_rows").as_array()) {
        gbe::SubframeRow row;
        for (const auto& kv : r.as_object()) {
            const auto& key = kv.first;
            // group column -> int64, else -> double
            bool is_gc = false;
            for (const auto& gc : b.schema.group_columns) {
                if (gc == key) { is_gc = true; break; }
            }
            if (is_gc) {
                row.group_values[key] = as_int64(kv.second);
            } else {
                row.coeff_values[key] = kv.second.as_number();
            }
        }
        b.rows.push_back(std::move(row));
    }

    // query_positions: interpret as integer compact indices for lookup.
    // (For linear in Turn 4, same field reused with float semantics.)
    for (const auto& qp : input.at("query_positions").as_array()) {
        std::vector<int64_t> one;
        for (const auto& v : qp.as_array()) {
            // Query positions are stored as floats in the fixture JSON.
            // For lookup, they are integer-valued; static_cast is fine.
            one.push_back(static_cast<int64_t>(v.as_number()));
        }
        b.query_positions.push_back(std::move(one));
    }

    // predictor_values_per_query
    for (const auto& pv : input.at("predictor_values_per_query").as_array()) {
        std::vector<double> one;
        for (const auto& v : pv.as_array()) {
            one.push_back(v.as_number());
        }
        b.predictor_values_per_query.push_back(std::move(one));
    }

    return b;
}

// Emit the structured JSON response.
void emit_success(const gbe::GroupByRegressionEvaluator& ev,
                  const FixtureBundle& b)
{
    // Evaluate all queries
    std::map<std::string, std::vector<double>> predictions;
    for (const auto& t : b.schema.targets) predictions[t] = {};

    for (std::size_t i = 0; i < b.query_positions.size(); ++i) {
        auto r = ev.evaluate_lookup(
            b.query_positions[i],
            b.predictor_values_per_query[i]);
        for (std::size_t ti = 0; ti < b.schema.targets.size(); ++ti) {
            predictions[b.schema.targets[ti]].push_back(r[ti]);
        }
    }

    std::ostringstream oss;
    oss << "{\"status\":\"ok\",\"predictions\":{";
    bool first_tgt = true;
    for (const auto& t : b.schema.targets) {
        if (!first_tgt) oss << ',';
        first_tgt = false;
        oss << '\"' << t << "\":[";
        const auto& vec = predictions[t];
        for (std::size_t i = 0; i < vec.size(); ++i) {
            if (i) oss << ',';
            oss << format_double(vec[i]);
        }
        oss << "]";
    }
    oss << "},\"actual_bin_centers\":[";
    for (std::size_t d = 0; d < ev.bin_centers().size(); ++d) {
        if (d) oss << ',';
        oss << format_int_array(ev.bin_centers()[d]);
    }
    oss << "],\"actual_valid_mask_flat\":"
        << format_bool_array(ev.valid_mask())
        << ",\"actual_grid_shape\":" << format_int_array(ev.grid_shape())
        << "}";
    std::cout << oss.str() << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cout << "{\"status\":\"error\",\"error\":"
                  << "\"usage: cli_runner <fixture.json>\"}" << std::endl;
        return 1;
    }
    try {
        const JsonValue root = gbe::parse_json_file(argv[1]);
        const FixtureBundle b = unpack_fixture(root);
        gbe::GroupByRegressionEvaluator ev(
            b.schema, b.rows, b.method, b.bounds);
        if (b.method != gbe::MethodMode::Lookup) {
            // Turn 3 scope: lookup only. Emit structured error.
            std::cout << "{\"status\":\"error\",\"error\":"
                      << "\"method 'linear' not implemented in Turn 3\"}"
                      << std::endl;
            return 2;
        }
        emit_success(ev, b);
        return 0;
    } catch (const std::exception& e) {
        // Escape quotes in the error message
        std::string msg = e.what();
        std::string escaped;
        for (char c : msg) {
            if (c == '"' || c == '\\') escaped.push_back('\\');
            escaped.push_back(c);
        }
        std::cout << "{\"status\":\"error\",\"error\":\""
                  << escaped << "\"}" << std::endl;
        return 3;
    }
}
