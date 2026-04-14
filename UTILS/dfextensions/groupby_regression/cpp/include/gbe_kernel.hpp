// cpp/include/gbe_kernel.hpp
//
// Phase 13.18.GB Layer A — GroupByRegressionEvaluator C++ kernel.
//
// Pure C++17, NO ROOT dependency, WASM-safe. This header is the
// canonical Layer A contract that ADF bridge + ROOT glue both target.
//
// IMPORTANT for consumers:
//   - All `position` arguments are COMPACT 0..N-1 INTEGER INDICES into
//     per-dimension `bin_centers`. They are NOT natural bin labels
//     (e.g. sector=5 must be remapped by the caller to its compact
//     index, typically via `remap()` below).
//   - Natural-label → compact-index remap at query time is the
//     consumer's responsibility (Phase 13.18.GB proposal v1.1 §3,
//     ADF review flag 3).
//   - Missing-bin Safety contract: any query whose compact index lands
//     on a cell with valid_mask=false returns NaN regardless of bounds
//     mode. No silent neighbour fallback. (Phase 13.18.GBADF v0.3 §4.4.)
//
// Turn 3: lookup. Turn 4: linear. Both implemented.

#ifndef GBE_KERNEL_HPP
#define GBE_KERNEL_HPP

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace gbe {

// BoundsMode mirrors the Python evaluator's bounds parameter.
enum class BoundsMode {
    Nan,     // out-of-grid -> NaN
    Clamp,   // out-of-grid -> snap to nearest in-grid index
};

// MethodMode mirrors the Python evaluator's method parameter.
// Turn 3 implements Lookup only; Linear lands in Turn 4.
enum class MethodMode {
    Lookup,
    Linear,
};

// Schema captured at load time from the dfGB (sparse rows + column
// naming conventions). Filled by the caller (JSON fixture reader in
// tests; ROOT TFile reader in Layer B Turn 6).
struct ModelSchema {
    std::vector<std::string> group_columns;       // index dimensions
    std::vector<std::string> predictor_columns;   // slope predictors
    std::vector<std::string> targets;             // target names
    std::string suffix;                           // coefficient suffix
    bool fit_intercept;                           // intercept present?
};

// Single populated subframe row, passed at construction.
// Maps column name -> value. Integer group_columns are stored in
// group_values; float coefficient values in coeff_values. Unknown-suffix
// columns (e.g. "_err_sw", "_rmse_sw", "_n_fitted_sw") are tolerated
// but not exposed through evaluate (Phase 13.18.GB P1-delta).
struct SubframeRow {
    std::map<std::string, int64_t> group_values;
    std::map<std::string, double>  coeff_values;
};

// Canonical Layer A contract.
//
// Constructor builds the dense N-D representation from sparse rows:
//   - Per-dim distinct sorted bin_centers inferred from data
//   - Per-dim natural-label -> compact-index remap built at load time
//   - Dense coefficient arrays allocated and populated
//   - valid_mask flat N-D bool filled (true where a row was present)
//
// Throws std::invalid_argument on structural errors (missing required
// coefficient columns, inconsistent row shapes, etc.).
class GroupByRegressionEvaluator {
public:
    GroupByRegressionEvaluator(ModelSchema schema,
                               std::vector<SubframeRow> rows,
                               MethodMode method,
                               BoundsMode bounds);

    // Evaluate a single query position. position is a compact index
    // per group_column, followed by predictor values (float) if any.
    //
    // position_idx must have size == schema.group_columns.size().
    // predictor_values must have size == schema.predictor_columns.size().
    //
    // Returns per-target predicted values in the order targets appear
    // in the schema.
    //
    // Turn 3: lookup only. If method_ == Linear this throws.
    std::vector<double> evaluate_lookup(
        const std::vector<int64_t>& position_idx,
        const std::vector<double>& predictor_values) const;

    // Evaluate at a fractional (float) position via N-D multilinear
    // interpolation over 2^N corners. Corners with valid_mask=false
    // are excluded from the weighted sum and remaining valid-corner
    // weights are renormalized to sum to 1. If ALL corners are
    // invalid, returns NaN per Safety contract (proposal v1.1 §6.3,
    // P1-β closure in Turn 5).
    //
    // bounds='nan': position outside [0, N-1] in any dim -> NaN
    // bounds='clamp': out-of-grid clamped to nearest in-grid edge
    // before interpolation.
    //
    // Turn 4: linear only. If method_ == Lookup this throws.
    std::vector<double> evaluate_linear(
        const std::vector<double>& position,
        const std::vector<double>& predictor_values) const;

    // -------- public accessors (ADF-facing) --------

    // Schema as provided at construction.
    const ModelSchema& schema() const noexcept { return schema_; }

    // Per-dim bin_centers (natural-label sorted distinct values).
    // bin_centers()[d] is the sorted vector of natural labels for
    // dimension d.
    const std::vector<std::vector<int64_t>>& bin_centers() const noexcept {
        return bin_centers_;
    }

    // Per-dim compact remap: remap()[d][natural_label] == compact_idx.
    // ADF bridge should READ from this (single source of truth) rather
    // than rebuilding its own — prevents load-time vs query-time drift
    // (ADF review flag: signal-loss incident).
    const std::vector<std::map<int64_t, int>>& remap() const noexcept {
        return remap_;
    }

    // Grid shape (per-dim size).
    const std::vector<int>& grid_shape() const noexcept { return grid_shape_; }

    // Flat valid_mask over the N-D grid (row-major linearization over
    // grid_shape_). True at positions that had a populated subframe row,
    // false at missing bins. Python parity: evaluator.valid_mask()
    // method. ADF review flag 2: symmetric naming across languages.
    const std::vector<bool>& valid_mask() const noexcept { return valid_mask_; }

    // Convenience: test whether a compact index tuple points at a
    // populated bin. Bounds-checks the index; returns false for any
    // out-of-grid index.
    bool is_valid_bin(const std::vector<int64_t>& position_idx) const noexcept;

    // Number of cells in the grid (product of grid_shape).
    int total_cells() const noexcept { return total_cells_; }

    // Number of populated (valid_mask=true) cells.
    int populated_cells() const noexcept { return populated_cells_; }

    // Method / bounds as configured.
    MethodMode method() const noexcept { return method_; }
    BoundsMode bounds() const noexcept { return bounds_; }

private:
    // Linearize a compact multi-index to flat offset. Assumes index is
    // already in-range per grid_shape_.
    int linearize_(const std::vector<int64_t>& compact_idx) const noexcept;

    // Apply bounds to an incoming compact index per-dim. Returns:
    //   - std::nullopt if any dim is out-of-grid AND bounds_ == Nan
    //   - otherwise a same-shape vector of in-grid indices (clamping
    //     per dim when bounds_ == Clamp, or original if already in-range)
    std::optional<std::vector<int64_t>> apply_bounds_(
        const std::vector<int64_t>& position_idx) const noexcept;

    // Coefficient column name builders (match Python's fit_intercept /
    // slope_<pred> convention).
    std::string intercept_col_(const std::string& target) const;
    std::string slope_col_(const std::string& target,
                           const std::string& predictor) const;

    // --------- state (immutable after construction) ---------
    ModelSchema schema_;
    MethodMode  method_;
    BoundsMode  bounds_;

    std::vector<std::vector<int64_t>> bin_centers_; // per-dim
    std::vector<std::map<int64_t, int>> remap_;     // per-dim natural->compact
    std::vector<int> grid_shape_;                   // per-dim size
    int total_cells_;
    int populated_cells_;

    std::vector<bool> valid_mask_;                  // flat row-major
    // Coefficient arrays: one flat vector per coefficient column name.
    std::map<std::string, std::vector<double>> coeff_arrays_;
};

} // namespace gbe

#endif // GBE_KERNEL_HPP
