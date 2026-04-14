// cpp/src/gbe_kernel.cpp
//
// Phase 13.18.GB Layer A implementation.
// Turn 3: constructor + evaluate_lookup. Linear arrives Turn 4.

#include "gbe_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <set>
#include <sstream>
#include <stdexcept>

namespace gbe {

namespace {

// Helper: build sorted unique values per group column.
std::vector<std::vector<int64_t>> infer_bin_centers(
    const ModelSchema& schema,
    const std::vector<SubframeRow>& rows)
{
    const std::size_t ndim = schema.group_columns.size();
    std::vector<std::set<int64_t>> per_dim(ndim);
    for (const auto& row : rows) {
        for (std::size_t d = 0; d < ndim; ++d) {
            const auto& gc = schema.group_columns[d];
            auto it = row.group_values.find(gc);
            if (it == row.group_values.end()) {
                std::ostringstream oss;
                oss << "row missing group column " << gc;
                throw std::invalid_argument(oss.str());
            }
            per_dim[d].insert(it->second);
        }
    }
    std::vector<std::vector<int64_t>> out(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        out[d].assign(per_dim[d].begin(), per_dim[d].end());
    }
    return out;
}

} // namespace

GroupByRegressionEvaluator::GroupByRegressionEvaluator(
    ModelSchema schema,
    std::vector<SubframeRow> rows,
    MethodMode method,
    BoundsMode bounds)
    : schema_(std::move(schema)),
      method_(method),
      bounds_(bounds),
      total_cells_(0),
      populated_cells_(0)
{
    if (schema_.group_columns.empty()) {
        throw std::invalid_argument(
            "schema.group_columns must be non-empty");
    }
    if (schema_.targets.empty()) {
        throw std::invalid_argument("schema.targets must be non-empty");
    }
    if (rows.empty()) {
        throw std::invalid_argument("rows must be non-empty");
    }

    // --- bin_centers + remap + grid_shape ---
    bin_centers_ = infer_bin_centers(schema_, rows);
    const std::size_t ndim = schema_.group_columns.size();
    remap_.resize(ndim);
    grid_shape_.resize(ndim);
    for (std::size_t d = 0; d < ndim; ++d) {
        const auto& bc = bin_centers_[d];
        grid_shape_[d] = static_cast<int>(bc.size());
        for (std::size_t i = 0; i < bc.size(); ++i) {
            remap_[d].emplace(bc[i], static_cast<int>(i));
        }
    }
    total_cells_ = 1;
    for (int n : grid_shape_) {
        total_cells_ *= n;
    }
    if (total_cells_ <= 0) {
        throw std::invalid_argument("degenerate grid_shape");
    }

    // --- allocate flat arrays ---
    valid_mask_.assign(static_cast<std::size_t>(total_cells_), false);
    const double kNaN = std::nan("");
    const auto ncells = static_cast<std::size_t>(total_cells_);

    // Enumerate required coefficient column names
    std::vector<std::string> coeff_names;
    for (const auto& t : schema_.targets) {
        if (schema_.fit_intercept) {
            coeff_names.push_back(intercept_col_(t));
        }
        for (const auto& p : schema_.predictor_columns) {
            coeff_names.push_back(slope_col_(t, p));
        }
    }
    for (const auto& name : coeff_names) {
        coeff_arrays_[name].assign(ncells, kNaN);
    }

    // --- populate ---
    for (const auto& row : rows) {
        // Build compact index
        std::vector<int64_t> compact(ndim);
        for (std::size_t d = 0; d < ndim; ++d) {
            const auto& gc = schema_.group_columns[d];
            auto it = row.group_values.find(gc);
            // it is guaranteed present (infer_bin_centers already checked)
            auto rem = remap_[d].find(it->second);
            if (rem == remap_[d].end()) {
                // Should not happen; bin_centers contain exactly the
                // values in rows by construction.
                std::ostringstream oss;
                oss << "internal: natural label " << it->second
                    << " not in remap[" << d << "]";
                throw std::logic_error(oss.str());
            }
            compact[d] = rem->second;
        }
        const int flat = linearize_(compact);
        valid_mask_[static_cast<std::size_t>(flat)] = true;
        for (const auto& name : coeff_names) {
            auto cit = row.coeff_values.find(name);
            if (cit == row.coeff_values.end()) {
                std::ostringstream oss;
                oss << "row missing required coefficient column "
                    << name;
                throw std::invalid_argument(oss.str());
            }
            coeff_arrays_[name][static_cast<std::size_t>(flat)] = cit->second;
        }
    }

    // Count populated cells for debugging/introspection
    for (bool v : valid_mask_) {
        if (v) ++populated_cells_;
    }
}

std::string GroupByRegressionEvaluator::intercept_col_(
    const std::string& target) const
{
    return target + "_intercept" + schema_.suffix;
}

std::string GroupByRegressionEvaluator::slope_col_(
    const std::string& target,
    const std::string& predictor) const
{
    return target + "_slope_" + predictor + schema_.suffix;
}

int GroupByRegressionEvaluator::linearize_(
    const std::vector<int64_t>& compact_idx) const noexcept
{
    // Row-major: flat = i0 * (N1*N2*...) + i1 * (N2*...) + ... + iN-1.
    int flat = 0;
    for (std::size_t d = 0; d < compact_idx.size(); ++d) {
        flat = flat * grid_shape_[d]
             + static_cast<int>(compact_idx[d]);
    }
    return flat;
}

std::optional<std::vector<int64_t>>
GroupByRegressionEvaluator::apply_bounds_(
    const std::vector<int64_t>& position_idx) const noexcept
{
    if (position_idx.size() != grid_shape_.size()) {
        return std::nullopt; // treat as NaN (caller will propagate)
    }
    std::vector<int64_t> out(position_idx.size());
    for (std::size_t d = 0; d < position_idx.size(); ++d) {
        const int n = grid_shape_[d];
        int64_t p = position_idx[d];
        if (p < 0 || p >= n) {
            if (bounds_ == BoundsMode::Nan) {
                return std::nullopt;
            }
            // Clamp
            if (p < 0) p = 0;
            if (p >= n) p = n - 1;
        }
        out[d] = p;
    }
    return out;
}

bool GroupByRegressionEvaluator::is_valid_bin(
    const std::vector<int64_t>& position_idx) const noexcept
{
    if (position_idx.size() != grid_shape_.size()) return false;
    for (std::size_t d = 0; d < position_idx.size(); ++d) {
        const int64_t p = position_idx[d];
        if (p < 0 || p >= grid_shape_[d]) return false;
    }
    const int flat = linearize_(position_idx);
    return valid_mask_[static_cast<std::size_t>(flat)];
}

std::vector<double> GroupByRegressionEvaluator::evaluate_lookup(
    const std::vector<int64_t>& position_idx,
    const std::vector<double>& predictor_values) const
{
    if (method_ != MethodMode::Lookup) {
        throw std::logic_error(
            "evaluate_lookup called but method != Lookup");
    }
    if (position_idx.size() != schema_.group_columns.size()) {
        std::ostringstream oss;
        oss << "position_idx size " << position_idx.size()
            << " != group_columns size " << schema_.group_columns.size();
        throw std::invalid_argument(oss.str());
    }
    if (predictor_values.size() != schema_.predictor_columns.size()) {
        std::ostringstream oss;
        oss << "predictor_values size " << predictor_values.size()
            << " != predictor_columns size "
            << schema_.predictor_columns.size();
        throw std::invalid_argument(oss.str());
    }

    const double kNaN = std::nan("");
    std::vector<double> result(schema_.targets.size(), kNaN);

    auto bounded = apply_bounds_(position_idx);
    if (!bounded.has_value()) {
        return result; // NaN for every target (out-of-grid + bounds=Nan)
    }
    const auto& compact = *bounded;
    const int flat = linearize_(compact);

    // Safety contract: missing bin -> NaN regardless of bounds mode.
    if (!valid_mask_[static_cast<std::size_t>(flat)]) {
        return result;
    }

    // Evaluate each target
    for (std::size_t ti = 0; ti < schema_.targets.size(); ++ti) {
        const auto& t = schema_.targets[ti];
        double v = 0.0;
        if (schema_.fit_intercept) {
            const auto& arr = coeff_arrays_.at(intercept_col_(t));
            v += arr[static_cast<std::size_t>(flat)];
        }
        for (std::size_t pi = 0; pi < schema_.predictor_columns.size(); ++pi) {
            const auto& arr = coeff_arrays_.at(
                slope_col_(t, schema_.predictor_columns[pi]));
            v += arr[static_cast<std::size_t>(flat)] * predictor_values[pi];
        }
        result[ti] = v;
    }
    return result;
}

} // namespace gbe
