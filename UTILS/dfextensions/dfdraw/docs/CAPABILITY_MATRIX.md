# Capability Matrix — dfdraw

**Generated:** 2026-04-09 09:59 UTC
**Phase:** 13.15.DF
**Generator:** `scripts/generate_capability_matrix.py`
**Sources:** `tests/feature_taxonomy.py` + `tests/test_layer_classification.py`

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 6 | 14% |
| ☑️ Smoke-only | 37 | 86% |
| 🧨 Broken | 0 | 0% |
| 📋 Planned | 0 | 0% |
| **Total features** | **43** | |
| **Total proof tests** | **152** | |
| **Invariance tests** | **21** | |

**Status key:**
- ✅ Verified — has at least one invariance test (A ≡ B check)
- ☑️ Smoke-only — tests pass but only check 'no crash'
- 🧨 Broken — at least one test failing
- 📋 Planned — no tests mapped yet

## Features

| Status | Feature | Pass | Fail |
|--------|---------|-----:|-----:|
| | **CORE** | | |
| ☑️ | **CORE.constructor** — DFDraw constructor and data normalization | 1 | 0 |
| ☑️ | **CORE.sampling** — Random sampling (reproducible) | 1 | 0 |
| | **PLOT** | | |
| ☑️ | **PLOT.histogram** — 1D histogram | 2 | 0 |
| ☑️ | **PLOT.scatter** — Scatter plot | 1 | 0 |
| ☑️ | **PLOT.profile** — Profile plot (mean per bin) | 1 | 0 |
| ☑️ | **PLOT.hist2d** — 2D histogram (density heatmap) | 1 | 0 |
| ☑️ | **PLOT.hexbin** — Hexbin plot (hexagonal binning) | 1 | 0 |
| | **PROFILE** | | |
| ☑️ | **PROFILE.return_data** — Profile data export (return_data=True) | 3 | 0 |
| ☑️ | **PROFILE.min_entries** — Minimum entries filter (min_entries=3) | 3 | 0 |
| ☑️ | **PROFILE.group_by_bins** — Auto-bin float group_by (bins/quantiles) | 4 | 0 |
| ☑️ | **PROFILE.sort_groups** — Sorted group order (negative-safe intervals) | 3 | 0 |
| ☑️ | **PROFILE.weights** — Weighted profile (column or expression) | 5 | 0 |
| | **TITLE** | | |
| ☑️ | **TITLE.auto_title** — Automatic title from plot parameters | 10 | 0 |
| | **SAME** | | |
| ✅ | **SAME.axes_reuse** — same=True reuses last axes | 7 | 0 |
| ☑️ | **SAME.auto_features** — same=True auto-color, auto-label, legend | 4 | 0 |
| ☑️ | **SAME.title_append** — same=True title append + subtitle merge | 4 | 0 |
| ✅ | **SAME.override** — same=True precedence (ax= wins, explicit overrides) | 3 | 0 |
| ✅ | **SAME.cross_method** — same=True across plot types (profile on hist2d) | 4 | 0 |
| | **BATCH** | | |
| ☑️ | **BATCH.dict_format** — draw_batch dict format (original) | 5 | 0 |
| ✅ | **BATCH.group_format** — draw_batch group format with defaults hierarchy | 6 | 0 |
| ☑️ | **BATCH.subplot_grid** — Subplot grid (ncols, layout, figsize, suptitle) | 7 | 0 |
| ☑️ | **BATCH.same_in_group** — same=True within batch groups | 2 | 0 |
| ☑️ | **BATCH.verbose** — Verbose levels (0/1/2) | 3 | 0 |
| ☑️ | **BATCH.save** — Batch save and close figures | 2 | 0 |
| | **STYLE** | | |
| ☑️ | **STYLE.predefined** — Predefined styles (default, publication, presentation) | 3 | 0 |
| ☑️ | **STYLE.custom** — Custom style dict and JSON persistence | 4 | 0 |
| | **STATS** | | |
| ☑️ | **STATS.default_fields** — Auto-detect default stats fields by plot type | 1 | 0 |
| ☑️ | **STATS.range_aware** — Range-aware statistics (range_x, range_y) | 3 | 0 |
| ☑️ | **STATS.robust** — Robust statistics (median, MAD) | 2 | 0 |
| | **ANNOT** | | |
| ☑️ | **ANNOT.statistics_box** — Statistics annotation box | 3 | 0 |
| ☑️ | **ANNOT.reference_overlay** — Reference function overlay (Gaussian, callable) | 4 | 0 |
| | **ADF** | | |
| ☑️ | **ADF.axis_titles** — Duck-typed axis titles from AliasDataFrame | 5 | 0 |
| | **PYARROW** | | |
| ✅ | **PYARROW.input** — PyArrow Table input support | 7 | 0 |
| | **FACET** | | |
| ☑️ | **FACET.grid** — Facet subplot grids (group_by + facet=True) | 1 | 0 |
| | **COMPAT** | | |
| ☑️ | **COMPAT.profile** — Profile backward compatibility | 2 | 0 |
| | **VECTOR** | | |
| ☑️ | **VECTOR.parse** — Bracket syntax parsing with paren-aware split | 8 | 0 |
| ☑️ | **VECTOR.dispatch** — Vector dispatch across profile/hist/scatter/draw | 5 | 0 |
| ☑️ | **VECTOR.fail_fast** — Fail-fast guards on hist2d/hexbin; per-pair for stats | 3 | 0 |
| ☑️ | **VECTOR.style_channels** — Vector + group_by style channel decomposition (P1-2) | 4 | 0 |
| ☑️ | **VECTOR.contract** — Return contract: stats_list, ylabel, auto_title | 3 | 0 |
| ☑️ | **VECTOR.color_cycle** — Color cycle continuity with outer same=True (GPT5 fix) | 3 | 0 |
| ☑️ | **VECTOR.adf_integration** — Vector through AliasDataFrame entry point (P0-3) | 1 | 0 |
| ✅ | **VECTOR.invariance** — Vector ≡ scalar same-loop semantic invariance (strong A≡B) | 7 | 0 |

---

*Auto-generated. ✅ = invariance test (A ≡ B). ☑️ = smoke only.*