# Capability Matrix — dfdraw

**Generated:** 2026-05-21 20:47 UTC
**Phase:** 13.15.DF
**Generator:** `scripts/generate_capability_matrix.py`
**Sources:** `tests/feature_taxonomy.py` + `tests/test_layer_classification.py`

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 47 | 45% |
| ☑️ Smoke-only | 57 | 54% |
| 🧨 Broken | 0 | 0% |
| 📋 Planned | 1 | 1% |
| **Total features** | **105** | |
| **Total proof tests** | **468** | |
| **Invariance tests** | **239** | |

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
| ✅ | **PROFILE.group_by_bins** — Auto-bin float group_by (bins/quantiles) | 5 | 0 |
| ☑️ | **PROFILE.sort_groups** — Sorted group order (negative-safe intervals) | 3 | 0 |
| ☑️ | **PROFILE.weights** — Weighted profile (column or expression) | 5 | 0 |
| | **TITLE** | | |
| ☑️ | **TITLE.auto_title** — Automatic title from plot parameters | 10 | 0 |
| | **SAME** | | |
| ✅ | **SAME.axes_reuse** — same=True reuses last axes | 7 | 0 |
| ✅ | **SAME.auto_features** — same=True auto-color, auto-label, legend | 5 | 0 |
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
| ✅ | **VECTOR.color_cycle** — Color cycle continuity with outer same=True (GPT5 fix) | 4 | 0 |
| ☑️ | **VECTOR.adf_integration** — Vector through AliasDataFrame entry point (P0-3) | 1 | 0 |
| ✅ | **VECTOR.invariance** — Vector ≡ scalar same-loop semantic invariance (strong A≡B) | 7 | 0 |
| ✅ | **VECTOR.kwarg_propagation** — Vector path forwards all scalar-mode kwargs (FIX1) | 7 | 0 |
| ☑️ | **VECTOR.groupby_polish** — Vector + group_by deduplicated legend, title, layout (FIX1) | 5 | 0 |
| ☑️ | **VECTOR.kwarg_surface** — Vector dispatch forwards named-parameter surface + facet guard (FIX1) | 6 | 0 |
| | **QUANTILE** | | |
| ☑️ | **QUANTILE.error_bars** — Quantile error_bars mode (asymmetric bars from symmetric pair) | 8 | 0 |
| ☑️ | **QUANTILE.band** — Quantile band mode (fill_between from symmetric triple) | 8 | 0 |
| ☑️ | **QUANTILE.central** — Quantile central= parameter (mean/median/both/none) | 7 | 0 |
| ☑️ | **QUANTILE.auto_detection** — Quantile mode auto-detection from list shape | 5 | 0 |
| ☑️ | **QUANTILE.style_keys** — Quantile style keys (band.alpha, band.hatch, error_bars.capsize, central_default) | 8 | 0 |
| ☑️ | **QUANTILE.parity** — Quantile determinism + backward-compat regression-lock | 6 | 0 |
| | **COMPAT** | | |
| ☑️ | **COMPAT.bool_expression** — Boolean expression input (==, !=, >, <, &, |, ~) on all plot functions | 11 | 0 |
| | **CHANNEL** | | |
| ☑️ | **CHANNEL.assignment** — Algorithm A: automatic visual-channel assignment for N data channels | 41 | 0 |
| ☑️ | **CHANNEL.nested_band** — Nested-band detection (>=2 symmetric pairs, central optional) and rendering | 5 | 0 |
| ☑️ | **CHANNEL.factored_legend** — Factored legend with section headers (sum-not-product entry count) | 4 | 0 |
| ✅ | **CHANNEL.selection_delta** — selection_delta channel (per-curve selection_vector) — AD-61, AD-66, AD-67 | 4 | 0 |
| ✅ | **CHANNEL.weights_delta** — weights_delta channel (per-curve weights_vector) — AD-61, AD-66, AD-67 | 4 | 0 |
| ✅ | **CHANNEL.compose_inner** — vector_compose='inner' — element-wise pairing (AD-62) | 4 | 0 |
| ✅ | **CHANNEL.compose_outer** — vector_compose='outer' — cross-product (AD-62) | 2 | 0 |
| ✅ | **CHANNEL.delta_facet_label** — Per-curve label management for selection_delta/weights_delta channels | 3 | 0 |
| | **DATA** | | |
| ☑️ | **DATA.nan_policy** — Optional NaN/inf filter with nan_policy parameter | 6 | 0 |
| ☑️ | **DATA.counters** — Stats dict counters: n_input, n_filtered, n_inf_*, n_nan_* | 5 | 0 |
| | **AUTORANGE** | | |
| ☑️ | **AUTORANGE.hybrid** — Hybrid autorange (outlier-aware: robust + minmax combined) | 5 | 0 |
| ☑️ | **AUTORANGE.minmax** — Minmax autorange strategy (backward compat preset) | 2 | 0 |
| ☑️ | **AUTORANGE.percentile** — Percentile autorange strategies (percentile_99, percentile_95) | 1 | 0 |
| ☑️ | **AUTORANGE.diagnostics** — Stats keys autorange_used + autorange_strategy (AD-77) | 1 | 0 |
| | **DATA** | | |
| ✅ | **HIST.weights** — hist() weights= column or expression — Phase 13.27 Commit 2 FIX1 | 5 | 0 |
| | **COLUMN_REF** | | |
| ✅ | **COLUMN_REF.validation** — Column-reference parameter validation (Class-2 actionable errors) — Phase 13.30 | 12 | 0 |
| | **FACET** | | |
| ✅ | **FACET.column_mode** — facet_by accepts DataFrame column name (AD-78) — Phase 13.31 | 12 | 0 |
| ✅ | **FACET.column_mode_binning** — facet_by_bins / facet_by_quantiles auto-binning of float column facets (AD-79) — Phase 13.32 Sub-fix 3 | 8 | 0 |
| | **QUANTILE** | | |
| ✅ | **PROFILE.quantiles_grouped** — Per-group quantile band/discrete rendering on profile() — Phase 13.32 Sub-fix 2 | 6 | 0 |
| | **FACET** | | |
| ✅ | **FACET.title_display_name** — Subplot titles show original facet_by name, never internal __dfdraw_facet_bin__ (BUG-001) — Phase 13.32 FIX1 | 1 | 0 |
| ✅ | **FACET.auto_title** — auto_title=True produces fig.suptitle in faceted mode (BUG-002) — Phase 13.32 FIX1 | 2 | 0 |
| ✅ | **FACET.numeric_bin_sort** — Facet bin panels in numeric order, not lexicographic (BUG-003) — Phase 13.32 FIX1 | 1 | 0 |
| | **NORMALIZE** | | |
| ✅ | **NORMALIZE.delta** — normalize='delta': v[0]-v[1] per bin with SEM error propagation — Phase 13.33 M1 | 3 | 0 |
| ✅ | **NORMALIZE.ratio** — normalize='ratio': v[0]/v[1] with delta-method error, zero-denom mask — Phase 13.33 M1 | 3 | 0 |
| ✅ | **NORMALIZE.log_ratio** — normalize='log_ratio': ln(v[0]/v[1]) with non-positive mean mask — Phase 13.33 M1 | 2 | 0 |
| ✅ | **NORMALIZE.pull** — normalize='pull': (v[0]-v[1])/sigma with +/-1sigma/+/-2sigma bands (AD-82) — Phase 13.33 M1 | 2 | 0 |
| ✅ | **NORMALIZE.callable** — normalize=callable: user-supplied f(stats_0, stats_1) -> (values, errors) — Phase 13.33 M1 | 2 | 0 |
| ✅ | **NORMALIZE.layout** — normalize_layout: overlay+diff (2-panel) vs diff_only (single panel) — Phase 13.33 M1 | 3 | 0 |
| ✅ | **NORMALIZE.sign_convention** — AD-80 sign convention: vector[0]=signal, vector[1]=reference; delta=signal-reference — Phase 13.33 M1 | 1 | 0 |
| ✅ | **NORMALIZE.single_y_convention** — Single-Y + selection_vector + normalize forces vector_compose='outer' internally (§6 directive) — Phase 13.33 M1 | 2 | 0 |
| ✅ | **NORMALIZE.backward_compat** — normalize=None preserves pre-Phase-13.33 behavior bit-identical — Phase 13.33 M1 | 1 | 0 |
| ✅ | **NORMALIZE.validation** — normalize input validation: wrong vector count, invalid mode, same=True conflict — Phase 13.33 M1 | 3 | 0 |
| ✅ | **NORMALIZE.group_by_compose** — group_by + normalize: per-group differential rendering — Phase 13.33 M2 | 3 | 0 |
| ✅ | **NORMALIZE.facet_by_compose** — facet_by + normalize: K x 2 grid with per-facet independent differential; facet_by_bins/_quantiles raises (NF.3 workaround-hint lock) — Phase 13.33 M2 + FIX1 | 3 | 0 |
| | **ROBUSTNESS** | | |
| 📋 | **ROBUSTNESS.median_mad_sigma** — central='median' must use MAD-sigma error bars (Phase 13.33 CRR §11 pre-existing inconsistency lock; xfail until source-side fix) — Phase 13.34 M2 | 0 | 0 |
| ✅ | **ROBUSTNESS.stats_schema** — Stats dict key contract per plot kind — locks against silent renames breaking ADF/RootInteractive — Phase 13.34 M2 | 3 | 0 |
| ✅ | **ROBUSTNESS.kwarg_composition** — Feature interaction tests — would have caught BUG-001/002/003 at delivery; locks 5 known kwarg interaction pairs — Phase 13.34 M2 | 5 | 0 |
| | **HIST** | | |
| ☑️ | **HIST.step_per_group_color** — histtype='step' renders distinct per-group edgecolors (BUG-014 closed; extends Phase 13.36 sentinel to edgecolor) — Phase 13.37.DF | 4 | 0 |
| | **PROFILE** | | |
| ☑️ | **PROFILE.float_group_by_guard** — profile() float group_by with no bins + nunique>20 raises ValueError with group_by_bins=N guidance (BUG-015; mirrors Phase 13.35 hist BUG-012) — Phase 13.37.DF | 2 | 0 |
| | **HIST** | | |
| ☑️ | **HIST.interval_sort_numeric** — pd.Interval group legend sorted by numeric .left (BUG-016; hasattr-guard extension of _interval_sort_key) — Phase 13.37.DF | 3 | 0 |
| ☑️ | **HIST.hist_errors** — Poisson error bar overlay (hist_errors=True): √n raw / √n/N probability / √n/(N·bw_i) density (per-bin); weighted Poisson via Σw²; zero-bin masking; ungrouped+bins=int safe; Phase 13.36 color sentinel preserved — Phase 13.37.DF | 10 | 0 |
| | **PROFILE** | | |
| ☑️ | **PROFILE_HIST.linestyle_cycle** — linestyle_cycle=True cycles per-group linestyles from channels.cycles.linestyle on profile() and hist(); user-explicit linestyle= wins via _ud_user_linestyle sentinel (extends Phase 13.36 Edit 17 pattern) — Phase 13.37.DF | 5 | 0 |
| | **FACET** | | |
| ✅ | **FACET.float_facet_by_guard** — _dispatch_faceted_render() float facet_by + no bins + nunique>20 raises ValueError with facet_by_bins=N / facet_by_quantiles=N guidance (BUG-017; third instance of BUG-012/BUG-015 float-guard class) — Phase 13.38.DF | 2 | 0 |
| | **SCATTER** | | |
| ✅ | **SCATTER.xerr_yerr** — scatter() xerr/yerr from column name or df.eval() expression. Render via ax.errorbar() when either provided; ax.scatter() otherwise (dispatch invariance). Three-tier NaN policy: raise on 100%, warn at >50%, silent zeroing at ≤50%. nanfrac in stats dict. Style keys: scatter.error_capsize=2, scatter.error_elinewidth=1.0, scatter.error_ecolor=None — Phase 13.38.DF | 9 | 0 |
| ✅ | **SCATTER.expression_color** — scatter() color= accepts df.eval() expression (e.g. color='abs(tgl)'). _process_color() dispatch reordered (CP0-1): None → array → column → fixed-color (to_rgba) → df.eval → terminal. Column-name check precedes to_rgba() to preserve backward compat for columns named after matplotlib colors ('b', 'r', 'k') — Phase 13.38.DF | 5 | 0 |
| ✅ | **SCATTER.expression_marker** — scatter() marker= accepts boolean df.eval() expression (e.g. marker='ncl > 100'); True → 's', False → 'o'. Per-point rendering via np.unique loop with label='_nolegend_' (no spurious legend entries) — Phase 13.38.DF | 2 | 0 |
| ✅ | **SCATTER.expr_compose** — scatter() expression color + expression marker composition: both encodings simultaneously on single-path scatter. Each marker subgroup carries its own colormap array — Phase 13.38.DF | 1 | 0 |
| | **PROFILE** | | |
| ✅ | **PROFILE.profile2d** — profile('z:y:x') → 2D mean heatmap via scipy.stats.binned_statistic_2d + ax.pcolormesh. Supports bins=[nx,ny] or bins=nx+bins2=ny, min_entries_2d=N masking, norm='log', colorbar+clabel. Dispatch in DFDraw.profile() at colon_count==2 (CP1-5: after _apply_selection/_apply_sampling, before _parse_expr). z/y/x accept column names or df.eval() expressions. Backward compat: 'y:x' (colon_count==1) unchanged — Phase 13.39.DF | 9 | 0 |
| ✅ | **PROFILE.time_axis** — profile() time_format= pre-conversion: x_data converted to matplotlib date numbers via mdates.date2num() before binning. CP1-4 auto-detect: datetime64 column dtype detected BEFORE astype(float) (else int64-nanosecond cast becomes ~1.7e15 → pd.to_datetime crashes 'year out of range'). DateFormatter / AutoDateFormatter applied post-render — Phase 13.39.DF | 5 | 0 |
| | **HIST** | | |
| ✅ | **HIST.time_axis** — hist() time_format= pre-conversion: x_data → matplotlib date numbers BEFORE ax.hist(). Post-hoc rewrite would be no-op for Patches (lesson from Phase 13.39 v1.0 P1). CP1-1 regression-lock: §9.TA.5 uses realistic timestamps (~1.7e9), as epoch-0 made both pre-conv (0.0) and raw (0) paths pass — Phase 13.39.DF | 1 | 0 |
| | **SCATTER** | | |
| ✅ | **SCATTER.time_axis** — scatter() time_format= pre-conversion: x_data → matplotlib date numbers BEFORE ax.scatter/ax.errorbar — Phase 13.39.DF | 1 | 0 |
| ✅ | **SCATTER.scatter3d** — draw('z:y:x', type='scatter3d') → 3D point cloud via mpl_toolkits.mplot3d. Reuses Phase 13.38 _process_color() + _process_size() unchanged. color=/size= accept column names or df.eval() expressions. elev=/azim= for ax.view_init(). Stats dict locks mean_x AND mean_y AND mean_z to 1e-9 (CP1-3). Scope boundaries: group_by + scatter3d raises (CP2-1); same=True onto non-3D axes raises (CP2-2). 'y:x' (colon!=2) with type='scatter3d' raises with actionable message — Phase 13.39.DF | 8 | 0 |

---

*Auto-generated. ✅ = invariance test (A ≡ B). ☑️ = smoke only.*