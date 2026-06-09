# Capability Matrix — dfdraw

**Generated:** 2026-06-09 11:18 UTC
**Phase:** PHASE_13_51_DF_END
**Generator:** `scripts/generate_capability_matrix.py`
**Sources:** `tests/feature_taxonomy.py` + `tests/test_layer_classification.py`

## Summary

| Status | Count | % |
|--------|------:|--:|
| ✅ Verified | 59 | 44% |
| ☑️ Smoke-only | 73 | 55% |
| 🧨 Broken | 0 | 0% |
| 📋 Planned | 1 | 1% |
| **Total features** | **133** | |
| **Total proof tests** | **632** | |
| **Invariance tests** | **356** | |
| **Visual tests** | **27** | |

**Status key:**
- ✅ Verified — has at least one passing invariance test (A ≡ B check)
- ☑️ Smoke-only — tests pass but only check 'no crash'
- 🧨 Broken — at least one test failing
- 📋 Planned — no tests mapped yet
- 👁 Visual — orthogonal: feature has ≥1 visual_primitive test (see HTML view)

## Features

| Status | Feature | Tests | Pass | Fail | Inv | Visual |
|--------|---------|------:|-----:|-----:|----:|-------:|
| | **ADF** | | | | | |
| ☑️ | **ADF.axis_titles** — Duck-typed axis titles from AliasDataFrame | 5 | 5 | 0 | 0 |  |
| | **ANNOT** | | | | | |
| ☑️ | **ANNOT.statistics_box** — Statistics annotation box | 3 | 3 | 0 | 0 |  |
| ☑️ | **ANNOT.reference_overlay** — Reference function overlay (Gaussian, callable) | 4 | 4 | 0 | 0 |  |
| | **API** | | | | | |
| ✅ | **API.kwarg_typo_guard** — Kwarg-typo guard (difflib did-you-mean at draw() entry) | 1 | 1 | 0 | 1 |  |
| | **AUTORANGE** | | | | | |
| ☑️ | **AUTORANGE.hybrid** — Hybrid autorange (outlier-aware: robust + minmax combined) | 5 | 5 | 0 | 0 |  |
| ☑️ | **AUTORANGE.minmax** — Minmax autorange strategy (backward compat preset) | 2 | 2 | 0 | 0 |  |
| ☑️ | **AUTORANGE.percentile** — Percentile autorange strategies (percentile_99, percentile_95) | 1 | 1 | 0 | 0 |  |
| ☑️ | **AUTORANGE.diagnostics** — Stats keys autorange_used + autorange_strategy (AD-77) | 1 | 1 | 0 | 0 |  |
| | **BATCH** | | | | | |
| ☑️ | **BATCH.dict_format** — draw_batch dict format (original) | 5 | 5 | 0 | 0 |  |
| ✅ | **BATCH.group_format** — draw_batch group format with defaults hierarchy | 6 | 6 | 0 | 1 |  |
| ☑️ | **BATCH.subplot_grid** — Subplot grid (ncols, layout, figsize, suptitle) | 7 | 7 | 0 | 0 |  |
| ☑️ | **BATCH.same_in_group** — same=True within batch groups | 2 | 2 | 0 | 0 |  |
| ☑️ | **BATCH.verbose** — Verbose levels (0/1/2) | 3 | 3 | 0 | 0 |  |
| ☑️ | **BATCH.save** — Batch save and close figures | 2 | 2 | 0 | 0 |  |
| | **CHANNEL** | | | | | |
| ☑️ | **CHANNEL.assignment** — Algorithm A: automatic visual-channel assignment for N data channels | 41 | 41 | 0 | 0 |  |
| ☑️ | **CHANNEL.nested_band** — Nested-band detection (>=2 symmetric pairs, central optional) and rendering | 5 | 5 | 0 | 0 |  |
| ☑️ | **CHANNEL.factored_legend** — Factored legend with section headers (sum-not-product entry count) | 4 | 4 | 0 | 0 |  |
| ✅ | **CHANNEL.selection_delta** — selection_delta channel (per-curve selection_vector) — AD-61, AD-66, AD-67 | 4 | 4 | 0 | 4 |  |
| ✅ | **CHANNEL.weights_delta** — weights_delta channel (per-curve weights_vector) — AD-61, AD-66, AD-67 | 4 | 4 | 0 | 4 |  |
| ✅ | **CHANNEL.compose_inner** — vector_compose='inner' — element-wise pairing (AD-62) | 4 | 4 | 0 | 4 |  |
| ✅ | **CHANNEL.compose_outer** — vector_compose='outer' — cross-product (AD-62) | 2 | 2 | 0 | 2 |  |
| ✅ | **CHANNEL.delta_facet_label** — Per-curve label management for selection_delta/weights_delta channels | 3 | 3 | 0 | 3 |  |
| | **COLUMN_REF** | | | | | |
| ✅ | **COLUMN_REF.validation** — Column-reference parameter validation (Class-2 actionable errors) — Phase 13.30 | 12 | 12 | 0 | 12 |  |
| | **COMPAT** | | | | | |
| ☑️ | **COMPAT.profile** — Profile backward compatibility | 2 | 2 | 0 | 0 |  |
| ☑️ | **COMPAT.bool_expression** — Boolean expression input (==, !=, >, <, &, |, ~) on all plot functions | 11 | 11 | 0 | 0 |  |
| | **CORE** | | | | | |
| ☑️ | **CORE.constructor** — DFDraw constructor and data normalization | 1 | 1 | 0 | 0 |  |
| ☑️ | **CORE.sampling** — Random sampling (reproducible) | 1 | 1 | 0 | 0 |  |
| | **DATA** | | | | | |
| ☑️ | **DATA.nan_policy** — Optional NaN/inf filter with nan_policy parameter | 6 | 6 | 0 | 0 |  |
| ☑️ | **DATA.counters** — Stats dict counters: n_input, n_filtered, n_inf_*, n_nan_* | 5 | 5 | 0 | 0 |  |
| ✅ | **HIST.weights** — hist() weights= column or expression — Phase 13.27 Commit 2 FIX1 | 5 | 5 | 0 | 5 |  |
| | **FACET** | | | | | |
| ☑️ | **FACET.grid** — Facet subplot grids (group_by + facet=True) | 1 | 1 | 0 | 0 |  |
| ✅ | **FACET.column_mode** — facet_by accepts DataFrame column name (AD-78) — Phase 13.31 | 12 | 12 | 0 | 12 |  |
| ✅ | **FACET.column_mode_binning** — facet_by_bins / facet_by_quantiles auto-binning of float column facets (AD-79) — Phase 13.32 Sub-fix 3 | 8 | 8 | 0 | 8 |  |
| ✅ | **FACET.title_display_name** — Subplot titles show original facet_by name, never internal __dfdraw_facet_bin__ (BUG-001) — Phase 13.32 FIX1 | 1 | 1 | 0 | 1 |  |
| ✅ | **FACET.auto_title** — auto_title=True produces fig.suptitle in faceted mode (BUG-002) — Phase 13.32 FIX1 | 2 | 2 | 0 | 2 |  |
| ✅ | **FACET.numeric_bin_sort** — Facet bin panels in numeric order, not lexicographic (BUG-003) — Phase 13.32 FIX1 | 1 | 1 | 0 | 1 |  |
| ✅ | **FACET.float_facet_by_guard** — _dispatch_faceted_render() float facet_by + no bins + nunique>20 raises ValueError with facet_by_bins=N / facet_by_quantiles=N guidance (BUG-017; third instance of BUG-012/BUG-015 float-guard class) — Phase 13.38.DF | 2 | 2 | 0 | 2 |  |
| ✅ | **FACET.list_grid** — facet_by accepts Union[str, List[str]] for 1D/2D/3D faceting. Convention LOCKED matching numpy/pandas (n_rows, n_cols, ...) shape: facet_by[0]=ROW (vertical within figure), facet_by[1]=COLUMN (horizontal within figure), facet_by[2]=FIGID (separate figures, one per value). facet_by[3+] raises NotImplementedError. 3D returns (List[Figure], List[axes_2d], List[stats_dict]) — DEVIATES from standard (fig, ax, stats) contract; documented prominently in inline help. New params: share_x/share_y ∈ {'all','row','col','none'} (within-figure axis sharing), share_across_figures: bool (3D global range lock). Per-plot-kind lock for share_across_figures (CP1-2): scatter locks x AND y; hist/profile locks x only (y auto-scales per figure to handle sparse-figID variance). New helpers: _normalize_facet_args, _to_mpl_share (symmetric {'all':True,'row':'row','col':'col','none':False} — v1.2 CP0-1 fix for Hard Constraint #3), _validate_share_axis_value, _resolve_facet_values (discrete or pd.cut/qcut Interval), _filter_facet_value (CP1-3 discrete vs binned), _compute_global_ranges. dfdraw is FIRST major plotting library with unified API where Nth faceting dimension generates separate figures (seaborn/ggplot2/plotly/altair all require manual loops). _validate_facet_by_binning guard for list input (v1.3 P1-A). Per-plot-kind dispatch: hist uses range= (matplotlib convention); profile uses range= which DFDraw.profile remaps to draw_profile's x_range= internally; scatter uses ax.set_xlim/set_ylim post-draw (no native range params); hist also locks ax.set_xlim post-draw (range= only locks bins, not axis xlim). Empty cell handling: '(no data)' diagnostic + stats={'n':0,'empty':True} — Phase 13.41.DF | 23 | 23 | 0 | 23 |  |
| | **FIT** | | | | | |
| ✅ | **FIT.inline** — Inline fits (fit= parameter on hist/profile/scatter/draw) | 41 | 41 | 0 | 41 |  |
| ✅ | **FIT.summary** — Summary fit — standalone table + params figure | 27 | 27 | 0 | 27 |  |
| ✅ | **FIT.root_aliases** — ROOT-convention aliases (fit='gaus', type='histo') | 2 | 2 | 0 | 2 |  |
| ☑️ | **FIT.display_names** — Display-name map for fit parameters (render-only short/Greek names) 👁 | 3 | 3 | 0 | 0 | 3 |
| ☑️ | **FIT.precision_modes** — Separate value/error precision keys + physics alignment mode 👁 | 2 | 2 | 0 | 0 | 2 |
| ☑️ | **FIT.textbox_kwargs_extensions** — fit_textbox_kwargs extensions: rename_params / value_format / error_format / precision_mode 👁 | 2 | 2 | 0 | 0 | 2 |
| | **HIST** | | | | | |
| ☑️ | **HIST.step_per_group_color** — histtype='step' renders distinct per-group edgecolors (BUG-014 closed; extends Phase 13.36 sentinel to edgecolor) — Phase 13.37.DF | 4 | 4 | 0 | 0 |  |
| ☑️ | **HIST.interval_sort_numeric** — pd.Interval group legend sorted by numeric .left (BUG-016; hasattr-guard extension of _interval_sort_key) — Phase 13.37.DF | 3 | 3 | 0 | 0 |  |
| ☑️ | **HIST.hist_errors** — Poisson error bar overlay (hist_errors=True): √n raw / √n/N probability / √n/(N·bw_i) density (per-bin); weighted Poisson via Σw²; zero-bin masking; ungrouped+bins=int safe; Phase 13.36 color sentinel preserved — Phase 13.37.DF | 10 | 10 | 0 | 0 |  |
| ✅ | **HIST.time_axis** — hist() time_format= pre-conversion: x_data → matplotlib date numbers BEFORE ax.hist(). Post-hoc rewrite would be no-op for Patches (lesson from Phase 13.39 v1.0 P1). CP1-1 regression-lock: §9.TA.5 uses realistic timestamps (~1.7e9), as epoch-0 made both pre-conv (0.0) and raw (0) paths pass — Phase 13.39.DF | 1 | 1 | 0 | 1 |  |
| ✅ | **HIST.cumulative** — hist() cumulative=True/-1/False — ROOT TH1::Draw('cumulative') equivalent. Three values: True (ascending CDF/ECDF), False (default, byte-identical backward compat), -1 (descending/survival, ROOT convention). matplotlib native cumulative= forwarded explicitly at 4 internal call sites (Phase 13.39 §2.2 lesson applied recursively: DFDraw.hist → draw_hist → _draw_hist_grouped → ax.hist; ALSO through _dispatch_faceted_render for facet_by composition). Composes with: norm='probability' (→ ECDF 0-1), group_by overlaid (per-group ECDFs), group_by stacked (CP2-1 regression lock for 3rd call site), facet_by (per-facet cumulative), histtype='step' (HEP-standard step ECDF). Correctness guard (M5): hist_errors+cumulative → NotImplementedError (Poisson per-bin errors are independent; cumulative counts are correlated). Vector dispatch [x,y] propagates cumulative correctly (Phase 13.16.DF FIX1 bug class lock) — Phase 13.40.DF | 10 | 10 | 0 | 10 |  |
| | **LEGEND** | | | | | |
| ✅ | **LEGEND.modes** — legend= polymorphic kwarg (bool|str|dict) + show_legend= bool parallel + four modes (all|none|shared|first) 👁 | 6 | 6 | 0 | 2 | 4 |
| | **META** | | | | | |
| ✅ | **META.capability_matrix** — capability matrix integrity (taxonomy resolves; coverage; HTML; no orphan visuals) | 4 | 4 | 0 | 4 |  |
| | **NORMALIZE** | | | | | |
| ✅ | **NORMALIZE.delta** — normalize='delta': v[0]-v[1] per bin with SEM error propagation — Phase 13.33 M1 | 3 | 3 | 0 | 3 |  |
| ✅ | **NORMALIZE.ratio** — normalize='ratio': v[0]/v[1] with delta-method error, zero-denom mask — Phase 13.33 M1 | 3 | 3 | 0 | 3 |  |
| ✅ | **NORMALIZE.log_ratio** — normalize='log_ratio': ln(v[0]/v[1]) with non-positive mean mask — Phase 13.33 M1 | 2 | 2 | 0 | 2 |  |
| ✅ | **NORMALIZE.pull** — normalize='pull': (v[0]-v[1])/sigma with +/-1sigma/+/-2sigma bands (AD-82) — Phase 13.33 M1 | 2 | 2 | 0 | 2 |  |
| ✅ | **NORMALIZE.callable** — normalize=callable: user-supplied f(stats_0, stats_1) -> (values, errors) — Phase 13.33 M1 | 2 | 2 | 0 | 2 |  |
| ✅ | **NORMALIZE.layout** — normalize_layout: overlay+diff (2-panel) vs diff_only (single panel) — Phase 13.33 M1 | 3 | 3 | 0 | 3 |  |
| ✅ | **NORMALIZE.sign_convention** — AD-80 sign convention: vector[0]=signal, vector[1]=reference; delta=signal-reference — Phase 13.33 M1 | 1 | 1 | 0 | 1 |  |
| ✅ | **NORMALIZE.single_y_convention** — Single-Y + selection_vector + normalize forces vector_compose='outer' internally (§6 directive) — Phase 13.33 M1 | 2 | 2 | 0 | 2 |  |
| ✅ | **NORMALIZE.backward_compat** — normalize=None preserves pre-Phase-13.33 behavior bit-identical — Phase 13.33 M1 | 1 | 1 | 0 | 1 |  |
| ✅ | **NORMALIZE.validation** — normalize input validation: wrong vector count, invalid mode, same=True conflict — Phase 13.33 M1 | 3 | 3 | 0 | 3 |  |
| ✅ | **NORMALIZE.group_by_compose** — group_by + normalize: per-group differential rendering — Phase 13.33 M2 | 3 | 3 | 0 | 3 |  |
| ✅ | **NORMALIZE.facet_by_compose** — facet_by + normalize: K x 2 grid with per-facet independent differential; facet_by_bins/_quantiles raises (NF.3 workaround-hint lock) — Phase 13.33 M2 + FIX1 | 3 | 3 | 0 | 3 |  |
| | **OVERLAY** | | | | | |
| ☑️ | **OVERLAY.engine_layers_form** — overlay() engine with explicit layers=[...] (density base + overlays on one shared axes) | 4 | 4 | 0 | 0 |  |
| ☑️ | **OVERLAY.engine_guards** — engine ValueError guards (empty, non-method type, non-allowed type, >1 density, non-density base, 3D) raise BEFORE any draw | 7 | 7 | 0 | 0 |  |
| ☑️ | **OVERLAY.string_sugar_desugar** — draw(type='A+B') string sugar desugars to overlay engine with §1.4 routing policy (named-param splice complete per v1.5.1 P1-B) | 5 | 5 | 0 | 0 |  |
| ☑️ | **OVERLAY.faceting_rejected** — faceting/share_* params rejected at engine + sugar level (Phase 13.53 deferred) | 2 | 2 | 0 | 0 |  |
| ☑️ | **OVERLAY.range_lock_zorder** — post-draw range lock holds after plt.draw(); base-before-overlay z-order | 2 | 2 | 0 | 0 |  |
| ☑️ | **OVERLAY.summary_fit_profile_layer** — summary_fit on profile overlay layer renders (Phase 13.51 + 13.52 interaction) | 1 | 1 | 0 | 0 |  |
| | **PLOT** | | | | | |
| ☑️ | **PLOT.histogram** — 1D histogram | 2 | 2 | 0 | 0 |  |
| ☑️ | **PLOT.scatter** — Scatter plot | 1 | 1 | 0 | 0 |  |
| ☑️ | **PLOT.profile** — Profile plot (mean per bin) | 1 | 1 | 0 | 0 |  |
| ☑️ | **PLOT.hist2d** — 2D histogram (density heatmap) | 1 | 1 | 0 | 0 |  |
| ☑️ | **PLOT.hexbin** — Hexbin plot (hexagonal binning) | 1 | 1 | 0 | 0 |  |
| | **PROFILE** | | | | | |
| ☑️ | **PROFILE.return_data** — Profile data export (return_data=True) | 3 | 3 | 0 | 0 |  |
| ☑️ | **PROFILE.min_entries** — Minimum entries filter (min_entries=3) | 3 | 3 | 0 | 0 |  |
| ✅ | **PROFILE.group_by_bins** — Auto-bin float group_by (bins/quantiles) | 5 | 5 | 0 | 1 |  |
| ☑️ | **PROFILE.sort_groups** — Sorted group order (negative-safe intervals) | 3 | 3 | 0 | 0 |  |
| ☑️ | **PROFILE.weights** — Weighted profile (column or expression) | 5 | 5 | 0 | 0 |  |
| ☑️ | **PROFILE.float_group_by_guard** — profile() float group_by with no bins + nunique>20 raises ValueError with group_by_bins=N guidance (BUG-015; mirrors Phase 13.35 hist BUG-012) — Phase 13.37.DF | 2 | 2 | 0 | 0 |  |
| ☑️ | **PROFILE_HIST.linestyle_cycle** — linestyle_cycle=True cycles per-group linestyles from channels.cycles.linestyle on profile() and hist(); user-explicit linestyle= wins via _ud_user_linestyle sentinel (extends Phase 13.36 Edit 17 pattern) — Phase 13.37.DF | 5 | 5 | 0 | 0 |  |
| ✅ | **PROFILE.profile2d** — profile('z:y:x') → 2D mean heatmap via scipy.stats.binned_statistic_2d + ax.pcolormesh. Supports bins=[nx,ny] or bins=nx+bins2=ny, min_entries_2d=N masking, norm='log', colorbar+clabel. Dispatch in DFDraw.profile() at colon_count==2 (CP1-5: after _apply_selection/_apply_sampling, before _parse_expr). z/y/x accept column names or df.eval() expressions. Backward compat: 'y:x' (colon_count==1) unchanged — Phase 13.39.DF | 9 | 9 | 0 | 9 |  |
| ✅ | **PROFILE.time_axis** — profile() time_format= pre-conversion: x_data converted to matplotlib date numbers via mdates.date2num() before binning. CP1-4 auto-detect: datetime64 column dtype detected BEFORE astype(float) (else int64-nanosecond cast becomes ~1.7e15 → pd.to_datetime crashes 'year out of range'). DateFormatter / AutoDateFormatter applied post-render — Phase 13.39.DF | 5 | 5 | 0 | 5 |  |
| | **PYARROW** | | | | | |
| ✅ | **PYARROW.input** — PyArrow Table input support | 7 | 7 | 0 | 5 |  |
| | **QUANTILE** | | | | | |
| ☑️ | **QUANTILE.error_bars** — Quantile error_bars mode (asymmetric bars from symmetric pair) | 8 | 8 | 0 | 0 |  |
| ☑️ | **QUANTILE.band** — Quantile band mode (fill_between from symmetric triple) | 8 | 8 | 0 | 0 |  |
| ☑️ | **QUANTILE.central** — Quantile central= parameter (mean/median/both/none) | 7 | 7 | 0 | 0 |  |
| ☑️ | **QUANTILE.auto_detection** — Quantile mode auto-detection from list shape | 8 | 8 | 0 | 0 |  |
| ☑️ | **QUANTILE.style_keys** — Quantile style keys (band.alpha, band.hatch, error_bars.capsize, central_default) | 8 | 8 | 0 | 0 |  |
| ☑️ | **QUANTILE.parity** — Quantile determinism + backward-compat regression-lock | 6 | 6 | 0 | 0 |  |
| ✅ | **PROFILE.quantiles_grouped** — Per-group quantile band/discrete rendering on profile() — Phase 13.32 Sub-fix 2 | 6 | 6 | 0 | 6 |  |
| | **RANGE** | | | | | |
| ✅ | **RANGE.scatter** — range= on scatter via shared 2D resolver | 4 | 4 | 0 | 4 |  |
| ✅ | **RANGE.scatter_filter** — range= on scatter removes out-of-range points (FIX1 semantic) | 1 | 1 | 0 | 1 |  |
| | **ROBUSTNESS** | | | | | |
| 📋 | **ROBUSTNESS.median_mad_sigma** — central='median' must use MAD-sigma error bars (Phase 13.33 CRR §11 pre-existing inconsistency lock; xfail until source-side fix) — Phase 13.34 M2 | 1 | 0 | 0 | 0 |  |
| ✅ | **ROBUSTNESS.stats_schema** — Stats dict key contract per plot kind — locks against silent renames breaking ADF/RootInteractive — Phase 13.34 M2 | 3 | 3 | 0 | 3 |  |
| ✅ | **ROBUSTNESS.kwarg_composition** — Feature interaction tests — would have caught BUG-001/002/003 at delivery; locks 5 known kwarg interaction pairs — Phase 13.34 M2 | 5 | 5 | 0 | 5 |  |
| | **SAME** | | | | | |
| ✅ | **SAME.axes_reuse** — same=True reuses last axes | 7 | 7 | 0 | 3 |  |
| ✅ | **SAME.auto_features** — same=True auto-color, auto-label, legend | 5 | 5 | 0 | 1 |  |
| ☑️ | **SAME.title_append** — same=True title append + subtitle merge | 4 | 4 | 0 | 0 |  |
| ✅ | **SAME.override** — same=True precedence (ax= wins, explicit overrides) | 3 | 3 | 0 | 1 |  |
| ✅ | **SAME.cross_method** — same=True across plot types (profile on hist2d) | 4 | 4 | 0 | 4 |  |
| | **SCATTER** | | | | | |
| ✅ | **SCATTER.xerr_yerr** — scatter() xerr/yerr from column name or df.eval() expression. Render via ax.errorbar() when either provided; ax.scatter() otherwise (dispatch invariance). Three-tier NaN policy: raise on 100%, warn at >50%, silent zeroing at ≤50%. nanfrac in stats dict. Style keys: scatter.error_capsize=2, scatter.error_elinewidth=1.0, scatter.error_ecolor=None — Phase 13.38.DF | 9 | 9 | 0 | 9 |  |
| ✅ | **SCATTER.expression_color** — scatter() color= accepts df.eval() expression (e.g. color='abs(tgl)'). _process_color() dispatch reordered (CP0-1): None → array → column → fixed-color (to_rgba) → df.eval → terminal. Column-name check precedes to_rgba() to preserve backward compat for columns named after matplotlib colors ('b', 'r', 'k') — Phase 13.38.DF | 5 | 5 | 0 | 5 |  |
| ✅ | **SCATTER.expression_marker** — scatter() marker= accepts boolean df.eval() expression (e.g. marker='ncl > 100'); True → 's', False → 'o'. Per-point rendering via np.unique loop with label='_nolegend_' (no spurious legend entries) — Phase 13.38.DF | 2 | 2 | 0 | 2 |  |
| ✅ | **SCATTER.expr_compose** — scatter() expression color + expression marker composition: both encodings simultaneously on single-path scatter. Each marker subgroup carries its own colormap array — Phase 13.38.DF | 1 | 1 | 0 | 1 |  |
| ✅ | **SCATTER.time_axis** — scatter() time_format= pre-conversion: x_data → matplotlib date numbers BEFORE ax.scatter/ax.errorbar — Phase 13.39.DF | 1 | 1 | 0 | 1 |  |
| ✅ | **SCATTER.scatter3d** — draw('z:y:x', type='scatter3d') → 3D point cloud via mpl_toolkits.mplot3d. Reuses Phase 13.38 _process_color() + _process_size() unchanged. color=/size= accept column names or df.eval() expressions. elev=/azim= for ax.view_init(). Stats dict locks mean_x AND mean_y AND mean_z to 1e-9 (CP1-3). Scope boundaries: group_by + scatter3d raises (CP2-1); same=True onto non-3D axes raises (CP2-2). 'y:x' (colon!=2) with type='scatter3d' raises with actionable message — Phase 13.39.DF | 8 | 8 | 0 | 8 |  |
| | **STATS** | | | | | |
| ☑️ | **STATS.default_fields** — Auto-detect default stats fields by plot type | 1 | 1 | 0 | 0 |  |
| ☑️ | **STATS.range_aware** — Range-aware statistics (range_x, range_y) | 3 | 3 | 0 | 0 |  |
| ☑️ | **STATS.robust** — Robust statistics (median, MAD) | 2 | 2 | 0 | 0 |  |
| | **STYLE** | | | | | |
| ☑️ | **STYLE.predefined** — Predefined styles (default, publication, presentation) | 3 | 3 | 0 | 0 |  |
| ☑️ | **STYLE.custom** — Custom style dict and JSON persistence | 4 | 4 | 0 | 0 |  |
| | **SUMMARY_FIT** | | | | | |
| ✅ | **SUMMARY_FIT.placement** — summary_fit.placement axis: figure (default) / subfigure / pad with GridSpec pre-planning 👁 | 4 | 4 | 0 | 1 | 3 |
| ☑️ | **SUMMARY_FIT.orientation** — summary_fit.orientation axis: row (default) / column (transpose) — honored by all placements (figure, pad, subfigure) 👁 | 2 | 2 | 0 | 0 | 2 |
| | **TITLE** | | | | | |
| ☑️ | **TITLE.auto_title** — Automatic title from plot parameters | 10 | 10 | 0 | 0 |  |
| ✅ | **TITLE.get_suptitle** — _get_suptitle public-API helper (mpl >= 3.8 + fallback) | 1 | 1 | 0 | 1 |  |
| | **VECTOR** | | | | | |
| ☑️ | **VECTOR.parse** — Bracket syntax parsing with paren-aware split | 8 | 8 | 0 | 0 |  |
| ☑️ | **VECTOR.dispatch** — Vector dispatch across profile/hist/scatter/draw | 5 | 5 | 0 | 0 |  |
| ☑️ | **VECTOR.fail_fast** — Fail-fast guards on hist2d/hexbin; per-pair for stats | 3 | 3 | 0 | 0 |  |
| ☑️ | **VECTOR.style_channels** — Vector + group_by style channel decomposition (P1-2) | 4 | 4 | 0 | 0 |  |
| ☑️ | **VECTOR.contract** — Return contract: stats_list, ylabel, auto_title | 3 | 3 | 0 | 0 |  |
| ✅ | **VECTOR.color_cycle** — Color cycle continuity with outer same=True (GPT5 fix) | 4 | 4 | 0 | 1 |  |
| ☑️ | **VECTOR.adf_integration** — Vector through AliasDataFrame entry point (P0-3) | 1 | 1 | 0 | 0 |  |
| ✅ | **VECTOR.invariance** — Vector ≡ scalar same-loop semantic invariance (strong A≡B) | 7 | 7 | 0 | 7 |  |
| ✅ | **VECTOR.kwarg_propagation** — Vector path forwards all scalar-mode kwargs (FIX1) | 7 | 7 | 0 | 7 |  |
| ☑️ | **VECTOR.groupby_polish** — Vector + group_by deduplicated legend, title, layout (FIX1) | 5 | 5 | 0 | 0 |  |
| ☑️ | **VECTOR.kwarg_surface** — Vector dispatch forwards named-parameter surface + facet guard (FIX1) | 6 | 6 | 0 | 0 |  |
| | **VISUAL** | | | | | |
| ☑️ | **VISUAL.data_bounds** — plotted points lie within axes limits (C-9 regression lock) 👁 | 1 | 1 | 0 | 0 | 1 |
| ☑️ | **VISUAL.cell_population** — every visible facet cell drew data; ragged-grid padding excluded 👁 | 2 | 2 | 0 | 0 | 2 |
| ☑️ | **VISUAL.artist_count** — per-cell data-series / legend count matches per-cell filtered groups 👁 | 2 | 2 | 0 | 0 | 2 |
| ☑️ | **VISUAL.color_distinct** — per-group colors distinct — color-cycle not reset (AD-37 class) 👁 | 2 | 2 | 0 | 0 | 2 |
| ☑️ | **VISUAL.facet_grid** — facet grid shape + shared-axis consistency (figure-derived) 👁 | 2 | 2 | 0 | 0 | 2 |
| ☑️ | **VISUAL.title** — suptitle populated, not duplicated per-cell (C-3); content (I-4) 👁 | 2 | 2 | 0 | 0 | 2 |

## Unmatched Tests (472)

472 tests pytest collected that no feature claims.
Grouped by test-file prefix.

<details><summary><code>test_adf_integration.py</code> (14)</summary>

- `test_adf_integration.py::TestBatchWithAxisTitles::test_batch_uses_axis_titles`
- `test_adf_integration.py::TestDFDrawDataSourceStorage::test_data_source_is_original`
- `test_adf_integration.py::TestDuckTypedAxisTitles::test_explicit_label_overrides_title`
- `test_adf_integration.py::TestDuckTypedAxisTitles::test_hexbin_uses_axis_titles`
- `test_adf_integration.py::TestDuckTypedAxisTitles::test_hist2d_uses_axis_titles`
- `test_adf_integration.py::TestDuckTypedAxisTitles::test_no_title_uses_default`
- `test_adf_integration.py::TestDuckTypedAxisTitles::test_partial_titles`
- `test_adf_integration.py::TestDuckTypedAxisTitles::test_profile_uses_axis_titles`
- `test_adf_integration.py::TestGetLabelMethod::test_get_label_without_data_source`
- `test_adf_integration.py::TestMockADF::test_df_attribute`
- `test_adf_integration.py::TestMockADF::test_get_unset_title`
- `test_adf_integration.py::TestMockADF::test_set_get_title`
- `test_adf_integration.py::TestWithoutDataSource::test_hist_without_data_source`
- `test_adf_integration.py::TestWithoutDataSource::test_scatter_without_data_source`

</details>

<details><summary><code>test_auto_title.py</code> (17)</summary>

- `test_auto_title.py::TestAutoTitleHexbin::test_auto_title_hexbin`
- `test_auto_title.py::TestAutoTitleParts::test_callable_selection_skipped`
- `test_auto_title.py::TestAutoTitleParts::test_expr_only`
- `test_auto_title.py::TestAutoTitleParts::test_expr_plus_group`
- `test_auto_title.py::TestAutoTitleParts::test_long_selection_truncated`
- `test_auto_title.py::TestAutoTitleProfile::test_auto_title_with_group`
- `test_auto_title.py::TestAutoTitleProfile::test_auto_title_with_selection`
- `test_auto_title.py::TestAutoTitleProfile::test_auto_title_with_weights`
- `test_auto_title.py::TestAutoTitleStyle::test_style_default`
- `test_auto_title.py::TestBuildAutoTitle::test_non_string_selection_skipped`
- `test_auto_title.py::TestBuildAutoTitle::test_none_selection`
- `test_auto_title.py::TestBuildAutoTitle::test_with_group`
- `test_auto_title.py::TestBuildAutoTitle::test_with_selection`
- `test_auto_title.py::TestBuildAutoTitle::test_with_weights`
- `test_auto_title.py::TestParseAutoTitleParts::test_all_string`
- `test_auto_title.py::TestParseAutoTitleParts::test_expr_plus_group`
- `test_auto_title.py::TestParseAutoTitleParts::test_false`

</details>

<details><summary><code>test_batch.py</code> (22)</summary>

- `test_batch.py::TestDrawBatchAutoDetect::test_auto_detect_hist_for_1d`
- `test_batch.py::TestDrawBatchAutoDetect::test_auto_detect_scatter_for_2d`
- `test_batch.py::TestDrawBatchAutoDetect::test_explicit_type_overrides`
- `test_batch.py::TestDrawBatchBasic::test_fig_ax_returned_when_not_saving`
- `test_batch.py::TestDrawBatchBasic::test_returns_dict_with_results`
- `test_batch.py::TestDrawBatchBasic::test_stats_populated`
- `test_batch.py::TestDrawBatchCloseFigures::test_close_figures_false_keeps_fig`
- `test_batch.py::TestDrawBatchDefaults::test_defaults_applied`
- `test_batch.py::TestDrawBatchErrorHandling::test_errors_collected`
- `test_batch.py::TestDrawBatchErrorHandling::test_raise_stops_on_error`
- `test_batch.py::TestDrawBatchErrorHandling::test_skip_continues_on_error`
- `test_batch.py::TestDrawBatchFileLoad::test_json_file`
- `test_batch.py::TestDrawBatchFileLoad::test_json_with_plots_key`
- `test_batch.py::TestDrawBatchFileLoad::test_unsupported_format_raises`
- `test_batch.py::TestDrawBatchFileLoad::test_yaml_file`
- `test_batch.py::TestDrawBatchMissingExpr::test_missing_expr_error`
- `test_batch.py::TestDrawBatchPlotTypes::test_invalid_type_error`
- `test_batch.py::TestDrawBatchSaveDir::test_correct_format`
- `test_batch.py::TestDrawBatchSaveDir::test_path_in_result`
- `test_batch.py::TestDrawBatchSaveDir::test_saves_files`
- `test_batch.py::TestDrawBatchVerbose::test_verbose_false_silent`
- `test_batch.py::TestDrawBatchVerbose::test_verbose_true_prints`

</details>

<details><summary><code>test_batch_groups.py</code> (1)</summary>

- `test_batch_groups.py::test_return_structure`

</details>

<details><summary><code>test_data_sanitize_autorange.py</code> (6)</summary>

- `test_data_sanitize_autorange.py::TestNanPolicy::test_nan_policy_raise_only_fires_when_invalid_present`
- `test_data_sanitize_autorange.py::TestPlotIntegration::test_hist1d_clean_data_autorange_diagnostics`
- `test_data_sanitize_autorange.py::TestPlotIntegration::test_hist1d_explicit_minmax_strategy`
- `test_data_sanitize_autorange.py::TestPlotIntegration::test_hist1d_explicit_numeric_range_strategy_explicit`
- `test_data_sanitize_autorange.py::TestPlotIntegration::test_hist2d_with_inf_does_not_crash_and_reports_counters`
- `test_data_sanitize_autorange.py::TestPlotIntegration::test_nan_policy_raise_propagates_through_hist`

</details>

<details><summary><code>test_drawer.py</code> (22)</summary>

- `test_drawer.py::TestColumnEvaluation::test_eval_complex_expression`
- `test_drawer.py::TestColumnEvaluation::test_eval_computed_expression`
- `test_drawer.py::TestColumnEvaluation::test_eval_direct_column`
- `test_drawer.py::TestColumnEvaluation::test_eval_invalid_raises`
- `test_drawer.py::TestDFDrawInit::test_init_invalid_type_raises`
- `test_drawer.py::TestDFDrawInit::test_init_with_alias_dataframe_like`
- `test_drawer.py::TestDFDrawInit::test_init_with_dataframe`
- `test_drawer.py::TestDFDrawInit::test_init_with_dict`
- `test_drawer.py::TestDrawDispatch::test_draw_2d_dispatches_to_scatter`
- `test_drawer.py::TestDrawDispatch::test_draw_explicit_type_profile`
- `test_drawer.py::TestDrawDispatch::test_draw_invalid_type_raises`
- `test_drawer.py::TestExpressionParsing::test_parse_1d_expr`
- `test_drawer.py::TestExpressionParsing::test_parse_2d_expr`
- `test_drawer.py::TestExpressionParsing::test_parse_expr_with_spaces`
- `test_drawer.py::TestExpressionParsing::test_parse_invalid_expr_raises`
- `test_drawer.py::TestSampling::test_sampling_larger_than_data_returns_all`
- `test_drawer.py::TestSampling::test_sampling_limits_size`
- `test_drawer.py::TestSampling::test_sampling_none_returns_all`
- `test_drawer.py::TestSelection::test_selection_boolean_mask`
- `test_drawer.py::TestSelection::test_selection_callable`
- `test_drawer.py::TestSelection::test_selection_none`
- `test_drawer.py::TestSelection::test_selection_string_query`

</details>

<details><summary><code>test_hexbin.py</code> (31)</summary>

- `test_hexbin.py::TestBasicHexbin::test_hexbin_requires_2d_expr`
- `test_hexbin.py::TestBasicHexbin::test_hexbin_stats_keys`
- `test_hexbin.py::TestBasicHexbin::test_hexbin_stats_values`
- `test_hexbin.py::TestHexbinColorScale::test_hexbin_vmin_vmax`
- `test_hexbin.py::TestHexbinColormap::test_hexbin_clabel`
- `test_hexbin.py::TestHexbinColormap::test_hexbin_colorbar_false`
- `test_hexbin.py::TestHexbinColormap::test_hexbin_colorbar_true`
- `test_hexbin.py::TestHexbinColormap::test_hexbin_custom_cmap`
- `test_hexbin.py::TestHexbinColormap::test_hexbin_default_cmap`
- `test_hexbin.py::TestHexbinComputedExpression::test_hexbin_computed_both`
- `test_hexbin.py::TestHexbinComputedExpression::test_hexbin_computed_y`
- `test_hexbin.py::TestHexbinExistingAxes::test_hexbin_on_existing_axes`
- `test_hexbin.py::TestHexbinFacet::test_hexbin_facet_top_k`
- `test_hexbin.py::TestHexbinFacet::test_hexbin_facet_true`
- `test_hexbin.py::TestHexbinFacet::test_hexbin_facet_with_gridsize`
- `test_hexbin.py::TestHexbinFunctionalAPI::test_functional_hexbin`
- `test_hexbin.py::TestHexbinFunctionalAPI::test_functional_hexbin_kwargs`
- `test_hexbin.py::TestHexbinGridsize::test_hexbin_gridsize_default`
- `test_hexbin.py::TestHexbinGridsize::test_hexbin_gridsize_large`
- `test_hexbin.py::TestHexbinGridsize::test_hexbin_gridsize_small`
- `test_hexbin.py::TestHexbinLabels::test_hexbin_custom_labels`
- `test_hexbin.py::TestHexbinLabels::test_hexbin_title`
- `test_hexbin.py::TestHexbinLabels::test_hexbin_xlabel_default`
- `test_hexbin.py::TestHexbinLabels::test_hexbin_ylabel_default`
- `test_hexbin.py::TestHexbinMincnt::test_hexbin_mincnt`
- `test_hexbin.py::TestHexbinNormalization::test_hexbin_norm_default`
- `test_hexbin.py::TestHexbinNormalization::test_hexbin_norm_log`
- `test_hexbin.py::TestHexbinSampling::test_hexbin_sampling`
- `test_hexbin.py::TestHexbinSelection::test_hexbin_callable_selection`
- `test_hexbin.py::TestHexbinSelection::test_hexbin_string_selection`
- `test_hexbin.py::TestHexbinStatsBox::test_hexbin_stats_box_show`

</details>

<details><summary><code>test_hist2d.py</code> (29)</summary>

- `test_hist2d.py::TestBasicHist2D::test_hist2d_requires_2d_expr`
- `test_hist2d.py::TestBasicHist2D::test_hist2d_stats_keys`
- `test_hist2d.py::TestBasicHist2D::test_hist2d_stats_values`
- `test_hist2d.py::TestHist2DBins::test_hist2d_bins_int`
- `test_hist2d.py::TestHist2DBins::test_hist2d_bins_list`
- `test_hist2d.py::TestHist2DBins::test_hist2d_bins_tuple`
- `test_hist2d.py::TestHist2DColorScale::test_hist2d_vmin_vmax`
- `test_hist2d.py::TestHist2DColormap::test_hist2d_clabel`
- `test_hist2d.py::TestHist2DColormap::test_hist2d_colorbar_false`
- `test_hist2d.py::TestHist2DColormap::test_hist2d_colorbar_true`
- `test_hist2d.py::TestHist2DColormap::test_hist2d_custom_cmap`
- `test_hist2d.py::TestHist2DColormap::test_hist2d_default_cmap`
- `test_hist2d.py::TestHist2DComputedExpression::test_hist2d_computed_both`
- `test_hist2d.py::TestHist2DComputedExpression::test_hist2d_computed_y`
- `test_hist2d.py::TestHist2DDrawDispatch::test_draw_hist2d_type`
- `test_hist2d.py::TestHist2DExistingAxes::test_hist2d_on_existing_axes`
- `test_hist2d.py::TestHist2DFunctionalAPI::test_functional_hist2d`
- `test_hist2d.py::TestHist2DFunctionalAPI::test_functional_hist2d_kwargs`
- `test_hist2d.py::TestHist2DLabels::test_hist2d_custom_labels`
- `test_hist2d.py::TestHist2DLabels::test_hist2d_title`
- `test_hist2d.py::TestHist2DLabels::test_hist2d_xlabel_default`
- `test_hist2d.py::TestHist2DLabels::test_hist2d_ylabel_default`
- `test_hist2d.py::TestHist2DNormalization::test_hist2d_norm_count`
- `test_hist2d.py::TestHist2DNormalization::test_hist2d_norm_density`
- `test_hist2d.py::TestHist2DNormalization::test_hist2d_norm_log`
- `test_hist2d.py::TestHist2DSampling::test_hist2d_sampling`
- `test_hist2d.py::TestHist2DSelection::test_hist2d_callable_selection`
- `test_hist2d.py::TestHist2DSelection::test_hist2d_string_selection`
- `test_hist2d.py::TestHist2DStatsBox::test_hist2d_stats_box_show`

</details>

<details><summary><code>test_hist_stats.py</code> (16)</summary>

- `test_hist_stats.py::TestHist2dProfileRobustStats::test_hist2d_robust_stats_per_axis`
- `test_hist_stats.py::TestHist2dProfileRobustStats::test_hist2d_stat_fields_quantiles`
- `test_hist_stats.py::TestHist2dProfileRobustStats::test_profile_robust_summary_stats`
- `test_hist_stats.py::TestRobustStatsAlwaysOn::test_base_stats_always_present`
- `test_hist_stats.py::TestRobustStatsAlwaysOn::test_mad_correctness`
- `test_hist_stats.py::TestRobustStatsAlwaysOn::test_mad_sigma_gaussian`
- `test_hist_stats.py::TestRobustStatsAlwaysOn::test_median_correctness`
- `test_hist_stats.py::TestRobustStatsAlwaysOn::test_robust_stats_always_present`
- `test_hist_stats.py::TestStatFieldsGroups::test_quantiles_not_present_by_default`
- `test_hist_stats.py::TestStatFieldsGroups::test_quantiles_present_on_request`
- `test_hist_stats.py::TestStatFieldsGroups::test_shape_present_on_request`
- `test_hist_stats.py::TestStatFieldsGroups::test_stat_fields_all`
- `test_hist_stats.py::TestStatFieldsGroups::test_stat_fields_invalid_raises`
- `test_hist_stats.py::TestStatFieldsGroups::test_stat_fields_list_combo`
- `test_hist_stats.py::TestVectorAndEdgeCases::test_robust_stats_edge_cases`
- `test_hist_stats.py::TestVectorAndEdgeCases::test_vector_stat_fields_passthrough`

</details>

<details><summary><code>test_histogram.py</code> (21)</summary>

- `test_histogram.py::TestBasicHistogram::test_hist_stats_keys`
- `test_histogram.py::TestBasicHistogram::test_hist_stats_values`
- `test_histogram.py::TestBasicHistogram::test_hist_with_bins`
- `test_histogram.py::TestHistogramComputedExpression::test_hist_complex_expression`
- `test_histogram.py::TestHistogramComputedExpression::test_hist_computed_expression`
- `test_histogram.py::TestHistogramExistingAxes::test_hist_on_existing_axes`
- `test_histogram.py::TestHistogramFunctionalAPI::test_functional_hist`
- `test_histogram.py::TestHistogramFunctionalAPI::test_functional_hist_with_kwargs`
- `test_histogram.py::TestHistogramGroupBy::test_hist_grouped`
- `test_histogram.py::TestHistogramGroupBy::test_hist_grouped_top_k`
- `test_histogram.py::TestHistogramLabels::test_hist_title`
- `test_histogram.py::TestHistogramLabels::test_hist_xlabel_custom`
- `test_histogram.py::TestHistogramLabels::test_hist_xlabel_default`
- `test_histogram.py::TestHistogramNormalization::test_hist_norm_density`
- `test_histogram.py::TestHistogramNormalization::test_hist_norm_probability`
- `test_histogram.py::TestHistogramRange::test_hist_range`
- `test_histogram.py::TestHistogramSampling::test_hist_sampling`
- `test_histogram.py::TestHistogramSelection::test_hist_with_callable_selection`
- `test_histogram.py::TestHistogramSelection::test_hist_with_string_selection`
- `test_histogram.py::TestHistogramStatsBox::test_hist_stats_box_custom_fields`
- `test_histogram.py::TestHistogramStatsBox::test_hist_stats_box_show`

</details>

<details><summary><code>test_phase_13_27_commit2_selection_weights.py</code> (40)</summary>

- `test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_7_1element_degrades_to_scalar`
- `test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_8_empty_list_raises`
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_1_facet_by_quartile_with_selection_vector`
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_2_facet_by_quartile_with_weights_vector`
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_3_facet_inner_compose_no_crash`
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_4_facet_by_bins_with_selection_vector`
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_5_column_mode_facet_by_with_selection_vector_AD78`
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_6_facet_by_quantiles_with_selection_vector_AD79`
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_1_profile_vector_path_idempotent`
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_2_profile_scalar_path_no_extra_assign`
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_3_hist_vector_path_idempotent`
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_4_scatter_vector_path_idempotent`
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_5_facet_dispatch_idempotent`
- `test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_1_per_curve_sanitize_stats_in_stats_list`
- `test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_2_nan_policy_filter_default_no_crash`
- `test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_3_nan_policy_warn_emits_warning`
- `test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_4_no_nan_data_no_sanitize_warning`
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_6_single_y_selection_vector_outer`
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_7_single_y_selection_vector_inner_raises_actionable`
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_8_no_fix1_userwarning_fires`
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_9_inner_raise_message_names_lengths`
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_WDH_4_single_x_weights_vector_outer`
- `test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_1_no_vec_iteration_indices_backward_compat`
- `test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_2_profile_no_vector_unchanged`
- `test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_3_hist_no_vector_unchanged`
- `test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_4_scatter_no_vector_unchanged`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_1_hist_accepts_selection_vector`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_2_hist_selection_vector_bins_shared`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Scatter::test_SDS_1_scatter_accepts_selection_vector`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Scatter::test_SDS_2_scatter_selection_with_facet_by`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_1_selection_weights_explicit_rule`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_2_5channel_refuses_without_facet`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_3_3channel_with_selection_delta_resolves`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_4_explicit_rules_count`
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_5_combination_with_quantiles`
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_1_hist_weights_vector_runs`
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_2_hist_weights_vector_with_global`
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_2_2ch_weights_vector_inner`
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_5_weights_categorical_kwarg_accepted`
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Scatter::test_WDS_2_scatter_weights_vector_silently_dropped`

</details>

<details><summary><code>test_phase_13_27_facet_refactor.py</code> (10)</summary>

- `test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_groupby_dispatches_correctly`
- `test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_invalid_channel_name_raises`
- `test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_quantiles_one_subplot_per_quantile`
- `test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_vector_creates_n_subplots`
- `test_phase_13_27_facet_refactor.py::TestFacetCapacity::test_facet_max_capacity_fires`
- `test_phase_13_27_facet_refactor.py::TestFacetCapacity::test_facet_max_warn_mode`
- `test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_existing_test_facetstar_unchanged`
- `test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_facet_true_eqivalent_to_facet_by_groupby`
- `test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_old_facet_profile_kwargs_preserved`
- `test_phase_13_27_facet_refactor.py::TestFacetSameTrueExclusion::test_facet_by_and_same_true_raises`

</details>

<details><summary><code>test_phase_13_28_df_fix1_autorange_style_keys.py</code> (9)</summary>

- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_combined_autorange_overrides`
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_k_robust_override`
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_strategy_override`
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_k_outlier_in_default_style`
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_k_robust_in_default_style`
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_percentile_in_default_style`
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_strategy_in_default_style`
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestNoRegressionInExistingStyleKeys::test_profile_marker_still_present`
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestNoRegressionInExistingStyleKeys::test_quantile_band_alpha_still_present`

</details>

<details><summary><code>test_phase_13_32_groupby_quantiles_facet.py</code> (5)</summary>

- `test_phase_13_32_groupby_quantiles_facet.py::TestRepro::test_in_126_overlay_with_quantiles`
- `test_phase_13_32_groupby_quantiles_facet.py::TestRepro::test_in_129_facet_with_groupby_bins`
- `test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_cap_still_fires_without_binning`
- `test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_with_bins_honors_n`
- `test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_with_quantiles_honors_n`

</details>

<details><summary><code>test_phase_13_34_df_fix1_bug010.py</code> (5)</summary>

- `test_phase_13_34_df_fix1_bug010.py::TestBUG010_NormalizeAutoTitle::test_NB_1_auto_title_normalize_single_curve`
- `test_phase_13_34_df_fix1_bug010.py::TestBUG010_NormalizeAutoTitle::test_NB_2_auto_title_normalize_grouped`
- `test_phase_13_34_df_fix1_bug010.py::TestBUG010_NormalizeAutoTitle::test_NB_3_auto_title_normalize_faceted`
- `test_phase_13_34_df_fix1_bug010.py::TestBUG010_NormalizeAutoTitle::test_NB_4_explicit_title_takes_precedence_over_auto_title`
- `test_phase_13_34_df_fix1_bug010.py::TestBUG010_NormalizeAutoTitle::test_NB_5_auto_title_false_no_suptitle`

</details>

<details><summary><code>test_phase_13_35_df_hist_groupby.py</code> (11)</summary>

- `test_phase_13_35_df_hist_groupby.py::TestHistGroupByBackwardCompat::test_HGBC_1_no_groupby_byte_identical`
- `test_phase_13_35_df_hist_groupby.py::TestHistGroupByBackwardCompat::test_HGBC_2_categorical_groupby_unchanged`
- `test_phase_13_35_df_hist_groupby.py::TestHistGroupByBins::test_HGB_1_group_by_bins_no_facet`
- `test_phase_13_35_df_hist_groupby.py::TestHistGroupByBins::test_HGB_2_group_by_bins_with_facet_by_architect_call`
- `test_phase_13_35_df_hist_groupby.py::TestHistGroupByBins::test_HGB_3_shared_bin_edges_across_groups`
- `test_phase_13_35_df_hist_groupby.py::TestHistGroupByStacked::test_HGSt_1_stacked_min_entries_alignment`
- `test_phase_13_35_df_hist_groupby.py::TestHistGroupByStats::test_HGS_1_n_groups_present_and_correct`
- `test_phase_13_35_df_hist_groupby.py::TestHistGroupByStats::test_HGS_2_bug012_float_high_cardinality_no_bins_raises`
- `test_phase_13_35_df_hist_groupby.py::TestHistNorm::test_HN_1_probability_sum_equals_one`
- `test_phase_13_35_df_hist_groupby.py::TestHistNorm::test_HN_2_default_none_preserves_raw_counts`
- `test_phase_13_35_df_hist_groupby.py::TestHistNorm::test_HN_3_density_integrates_to_one`

</details>

<details><summary><code>test_phase_13_36_df_user_style_override.py</code> (10)</summary>

- `test_phase_13_36_df_user_style_override.py::TestHistStyleOverride::test_SOH1_marker_warns_no_effect`
- `test_phase_13_36_df_user_style_override.py::TestHistStyleOverride::test_SOH2_color_uniform_step_histtype`
- `test_phase_13_36_df_user_style_override.py::TestLinestyleDocumentationLock::test_LS1_linestyle_None_group_by`
- `test_phase_13_36_df_user_style_override.py::TestScatterUntouched::test_SC1_scatter_color_marker_still_work`
- `test_phase_13_36_df_user_style_override.py::TestUserStyleOverride::test_SO1_marker_override_uniform`
- `test_phase_13_36_df_user_style_override.py::TestUserStyleOverride::test_SO2_color_override_with_warning`
- `test_phase_13_36_df_user_style_override.py::TestUserStyleOverride::test_SO3_markersize_override`
- `test_phase_13_36_df_user_style_override.py::TestUserStyleOverride::test_SO4_default_cycle_preserved`
- `test_phase_13_36_df_user_style_override.py::TestUserStyleOverride::test_SO5_bug013_same_true_marker_overlay`
- `test_phase_13_36_df_user_style_override.py::TestVectorPathForwarding::test_VF1_marker_via_vector_expression`

</details>

<details><summary><code>test_phase_13_51_post_audit.py</code> (23)</summary>

- `test_phase_13_51_post_audit.py::test_T10a_hist_datetime64_no_crash_scalarformatter_sentinel`
- `test_phase_13_51_post_audit.py::test_T10b_compute_autorange_datetime64_robust_3mad`
- `test_phase_13_51_post_audit.py::test_T10c_compute_autorange_datetime64_percentile_99`
- `test_phase_13_51_post_audit.py::test_T11_profile2d_wrapper_alias_for_profile`
- `test_phase_13_51_post_audit.py::test_T12_scatter3d_wrapper_and_type_collision_guard`
- `test_phase_13_51_post_audit.py::test_T13_draw_type_hexbin_dispatches_correctly`
- `test_phase_13_51_post_audit.py::test_T14_hist2d_time_format_auto_date_tick_labels`
- `test_phase_13_51_post_audit.py::test_T15_hist2d_fit_raises_clean_value_error`
- `test_phase_13_51_post_audit.py::test_T16_hexbin_range_raises_clean_value_error`
- `test_phase_13_51_post_audit.py::test_T16b_draw_type_hexbin_range_raises_clean_value_error_via_dispatch`
- `test_phase_13_51_post_audit.py::test_T17_hexbin_facet_by_raises_clean_value_error`
- `test_phase_13_51_post_audit.py::test_T17b_draw_type_hexbin_facet_by_raises_clean_value_error_via_dispatch`
- `test_phase_13_51_post_audit.py::test_T1_draw_type_profile_normalize_delta_yields_two_axes`
- `test_phase_13_51_post_audit.py::test_T2_draw_type_profile_facet_by_sec_populated_panels`
- `test_phase_13_51_post_audit.py::test_T3_draw_type_profile_selection_vector_matches_direct`
- `test_phase_13_51_post_audit.py::test_T4_draw_type_hist_facet_by_sec_populated_panels`
- `test_phase_13_51_post_audit.py::test_T5_draw_type_hist_selection_vector_matches_direct`
- `test_phase_13_51_post_audit.py::test_T6_draw_type_scatter_facet_by_sec_populated_panels`
- `test_phase_13_51_post_audit.py::test_T7_draw_type_scatter_time_format_auto_dateformatter`
- `test_phase_13_51_post_audit.py::test_T8_draw_type_hist2d_facet_by_sec_populated_panels`
- `test_phase_13_51_post_audit.py::test_T9a_profile2d_central_median_mesh_differs_from_mean`
- `test_phase_13_51_post_audit.py::test_T9b_profile_1d_central_median_line_differs_from_mean`
- `test_phase_13_51_post_audit.py::test_T9c_profile_fit_central_median_fit_center_differs`

</details>

<details><summary><code>test_phase_13_52_df_overlay.py</code> (1)</summary>

- `test_phase_13_52_df_overlay.py::test_T1p_overlay_profile2d_profile_alternative_density_base`

</details>

<details><summary><code>test_profile.py</code> (24)</summary>

- `test_profile.py::TestBasicProfile::test_profile_requires_2d_expr`
- `test_profile.py::TestBasicProfile::test_profile_stats_keys`
- `test_profile.py::TestBasicProfile::test_profile_stats_values`
- `test_profile.py::TestProfileBins::test_profile_bins_parameter`
- `test_profile.py::TestProfileBins::test_profile_range_parameter`
- `test_profile.py::TestProfileComputedExpression::test_profile_computed_both`
- `test_profile.py::TestProfileComputedExpression::test_profile_computed_y`
- `test_profile.py::TestProfileDrawDispatch::test_draw_profile_type`
- `test_profile.py::TestProfileErrorTypes::test_profile_error_none`
- `test_profile.py::TestProfileErrorTypes::test_profile_error_sem`
- `test_profile.py::TestProfileErrorTypes::test_profile_error_std`
- `test_profile.py::TestProfileExistingAxes::test_profile_on_existing_axes`
- `test_profile.py::TestProfileFunctionalAPI::test_functional_profile`
- `test_profile.py::TestProfileFunctionalAPI::test_functional_profile_kwargs`
- `test_profile.py::TestProfileGroupBy::test_profile_grouped`
- `test_profile.py::TestProfileGroupBy::test_profile_grouped_top_k`
- `test_profile.py::TestProfileLabels::test_profile_custom_labels`
- `test_profile.py::TestProfileLabels::test_profile_title`
- `test_profile.py::TestProfileLabels::test_profile_xlabel_default`
- `test_profile.py::TestProfileLabels::test_profile_ylabel_default`
- `test_profile.py::TestProfileSampling::test_profile_sampling`
- `test_profile.py::TestProfileSelection::test_profile_callable_selection`
- `test_profile.py::TestProfileSelection::test_profile_string_selection`
- `test_profile.py::TestProfileStatsBox::test_profile_stats_box_show`

</details>

<details><summary><code>test_pyarrow_input.py</code> (25)</summary>

- `test_pyarrow_input.py::TestBackendProperty::test_backend_pandas`
- `test_pyarrow_input.py::TestBatchWithPyArrow::test_batch_pyarrow`
- `test_pyarrow_input.py::TestChunkedArrays::test_chunked_array_handling`
- `test_pyarrow_input.py::TestComputedExpressions::test_computed_expression_parity`
- `test_pyarrow_input.py::TestComputedExpressions::test_hist_computed_expression`
- `test_pyarrow_input.py::TestComputedExpressions::test_scatter_computed_expression`
- `test_pyarrow_input.py::TestDrawDispatch::test_draw_hist_pyarrow`
- `test_pyarrow_input.py::TestDrawDispatch::test_draw_profile_pyarrow`
- `test_pyarrow_input.py::TestDrawDispatch::test_draw_scatter_pyarrow`
- `test_pyarrow_input.py::TestGroupByWithPyArrow::test_group_by_results_match`
- `test_pyarrow_input.py::TestGroupByWithPyArrow::test_group_by_works`
- `test_pyarrow_input.py::TestMemoryInfo::test_memory_info_keys_consistent`
- `test_pyarrow_input.py::TestMemoryInfo::test_memory_info_pandas`
- `test_pyarrow_input.py::TestMemoryInfo::test_memory_info_pyarrow`
- `test_pyarrow_input.py::TestOriginalTablePreserved::test_table_attribute`
- `test_pyarrow_input.py::TestOriginalTablePreserved::test_table_none_for_pandas`
- `test_pyarrow_input.py::TestPandasBackwardCompatibility::test_data_source_preserved`
- `test_pyarrow_input.py::TestPandasBackwardCompatibility::test_df_attribute_pandas`
- `test_pyarrow_input.py::TestPandasBackwardCompatibility::test_df_attribute_pyarrow`
- `test_pyarrow_input.py::TestPyArrowInputAcceptance::test_init_dict`
- `test_pyarrow_input.py::TestPyArrowInputAcceptance::test_init_invalid_type`
- `test_pyarrow_input.py::TestPyArrowInputAcceptance::test_init_pandas_dataframe`
- `test_pyarrow_input.py::TestPyArrowNotAvailable::test_pandas_works_without_pyarrow`
- `test_pyarrow_input.py::TestSelectionWithPyArrow::test_selection_results_match`
- `test_pyarrow_input.py::TestSelectionWithPyArrow::test_selection_works`

</details>

<details><summary><code>test_quantiles_profile.py</code> (37)</summary>

- `test_quantiles_profile.py::TestQuantileCentralLine::test_central_invalid_value_raises_valueerror`
- `test_quantiles_profile.py::TestQuantileDeterminism::test_det_band_both`
- `test_quantiles_profile.py::TestQuantileDeterminism::test_det_band_median`
- `test_quantiles_profile.py::TestQuantileDeterminism::test_det_band_none`
- `test_quantiles_profile.py::TestQuantileDeterminism::test_det_error_bars_median`
- `test_quantiles_profile.py::TestQuantileDeterminism::test_det_error_none_quantile`
- `test_quantiles_profile.py::TestQuantileDeterminism::test_det_with_groupby`
- `test_quantiles_profile.py::TestQuantileDeterminism::test_det_with_vector`
- `test_quantiles_profile.py::TestQuantileDocstrings::test_central_default_is_mean`
- `test_quantiles_profile.py::TestQuantileDocstrings::test_central_example`
- `test_quantiles_profile.py::TestQuantileDocstrings::test_mode_example`
- `test_quantiles_profile.py::TestQuantileDocstrings::test_quantiles_example`
- `test_quantiles_profile.py::TestQuantileErrorBarsRendering::test_error_bars_capsize_from_style`
- `test_quantiles_profile.py::TestQuantileErrorBarsRendering::test_error_bars_color_matches_central_line`
- `test_quantiles_profile.py::TestQuantileErrorBarsRendering::test_quantile_capsize_independent_of_profile_capsize`
- `test_quantiles_profile.py::TestQuantileErrorKwargInteraction::test_band_preserves_error_kwarg`
- `test_quantiles_profile.py::TestQuantileErrorKwargInteraction::test_default_error_rebinds_for_error_bars`
- `test_quantiles_profile.py::TestQuantileErrorKwargInteraction::test_error_quantile_without_quantiles_raises`
- `test_quantiles_profile.py::TestQuantileErrorKwargInteraction::test_explicit_none_renders_quantile_only`
- `test_quantiles_profile.py::TestQuantileErrorKwargInteraction::test_explicit_sem_renders_both`
- `test_quantiles_profile.py::TestQuantileGroupByInteraction::test_band_per_group`
- `test_quantiles_profile.py::TestQuantileGroupByInteraction::test_error_bars_per_group`
- `test_quantiles_profile.py::TestQuantileGroupByInteraction::test_grouped_legend_no_quantile_values`
- `test_quantiles_profile.py::TestQuantileGroupByInteraction::test_per_group_quantile_correctness`
- `test_quantiles_profile.py::TestQuantilePerBinCorrectness::test_per_bin_quantiles_handle_constant_data`
- `test_quantiles_profile.py::TestQuantilePerBinCorrectness::test_per_bin_quantiles_handle_empty_bin`
- `test_quantiles_profile.py::TestQuantilePerBinCorrectness::test_per_bin_quantiles_handle_n_equals_1`
- `test_quantiles_profile.py::TestQuantilePerBinCorrectness::test_per_bin_quantiles_ignore_nans`
- `test_quantiles_profile.py::TestQuantilePerBinCorrectness::test_per_bin_quantiles_with_weights_raises_notimplementederror_phaseb`
- `test_quantiles_profile.py::TestQuantileSameTrueLastAxContinuity::test_adf_cached_last_ax`
- `test_quantiles_profile.py::TestQuantileSameTrueLastAxContinuity::test_band_overlay_on_same`
- `test_quantiles_profile.py::TestQuantileSameTrueLastAxContinuity::test_last_ax_preserved`
- `test_quantiles_profile.py::TestQuantileSameTrueLastAxContinuity::test_new_call_creates_new_axes`
- `test_quantiles_profile.py::TestQuantileSameTrueLastAxContinuity::test_same_true_reuses_axes`
- `test_quantiles_profile.py::TestQuantileVectorInteraction::test_vector_band`
- `test_quantiles_profile.py::TestQuantileVectorInteraction::test_vector_error_bars`
- `test_quantiles_profile.py::TestQuantileVectorInteraction::test_vector_groupby_quantiles`

</details>

<details><summary><code>test_scatter.py</code> (29)</summary>

- `test_scatter.py::TestBasicScatter::test_scatter_requires_2d_expr`
- `test_scatter.py::TestBasicScatter::test_scatter_stats_keys`
- `test_scatter.py::TestBasicScatter::test_scatter_stats_values`
- `test_scatter.py::TestScatterColorMapping::test_scatter_color_column`
- `test_scatter.py::TestScatterColorMapping::test_scatter_colorbar_false`
- `test_scatter.py::TestScatterColorMapping::test_scatter_fixed_color`
- `test_scatter.py::TestScatterComputedExpression::test_scatter_computed_both`
- `test_scatter.py::TestScatterComputedExpression::test_scatter_computed_y`
- `test_scatter.py::TestScatterExistingAxes::test_scatter_on_existing_axes`
- `test_scatter.py::TestScatterFunctionalAPI::test_functional_scatter`
- `test_scatter.py::TestScatterFunctionalAPI::test_functional_scatter_kwargs`
- `test_scatter.py::TestScatterGroupBy::test_scatter_grouped`
- `test_scatter.py::TestScatterGroupBy::test_scatter_grouped_top_k`
- `test_scatter.py::TestScatterJitter::test_scatter_jitter_bool`
- `test_scatter.py::TestScatterJitter::test_scatter_jitter_float`
- `test_scatter.py::TestScatterJitter::test_scatter_jitter_tuple`
- `test_scatter.py::TestScatterLabels::test_scatter_custom_labels`
- `test_scatter.py::TestScatterLabels::test_scatter_title`
- `test_scatter.py::TestScatterLabels::test_scatter_xlabel_default`
- `test_scatter.py::TestScatterLabels::test_scatter_ylabel_default`
- `test_scatter.py::TestScatterMarkers::test_scatter_alpha`
- `test_scatter.py::TestScatterMarkers::test_scatter_custom_marker`
- `test_scatter.py::TestScatterSampling::test_scatter_sampling`
- `test_scatter.py::TestScatterSelection::test_scatter_callable_selection`
- `test_scatter.py::TestScatterSelection::test_scatter_string_selection`
- `test_scatter.py::TestScatterSizeMapping::test_scatter_fixed_size`
- `test_scatter.py::TestScatterSizeMapping::test_scatter_size_column`
- `test_scatter.py::TestScatterStatsBox::test_scatter_stats_box_custom`
- `test_scatter.py::TestScatterStatsBox::test_scatter_stats_box_show`

</details>

<details><summary><code>test_stats_enhancements.py</code> (41)</summary>

- `test_stats_enhancements.py::TestBreakingChange_ddof::test_std_population_vs_sample_difference`
- `test_stats_enhancements.py::TestBreakingChange_ddof::test_std_uses_population_ddof`
- `test_stats_enhancements.py::TestBreakingChange_ddof::test_std_x_uses_population_ddof`
- `test_stats_enhancements.py::TestBreakingChange_ddof::test_std_y_uses_population_ddof`
- `test_stats_enhancements.py::TestComputeStats::test_group_by_returns_multiple_rows`
- `test_stats_enhancements.py::TestComputeStats::test_range_passed_through`
- `test_stats_enhancements.py::TestComputeStats::test_returns_dataframe`
- `test_stats_enhancements.py::TestComputeStats::test_robust_passed_through`
- `test_stats_enhancements.py::TestIssue1_DefaultFieldsAdapt::test_explicit_fields_override_defaults`
- `test_stats_enhancements.py::TestIssue1_DefaultFieldsAdapt::test_format_stats_box_fallback_2d_detection`
- `test_stats_enhancements.py::TestIssue1_DefaultFieldsAdapt::test_format_stats_box_hexbin_defaults`
- `test_stats_enhancements.py::TestIssue1_DefaultFieldsAdapt::test_format_stats_box_hist2d_defaults`
- `test_stats_enhancements.py::TestIssue1_DefaultFieldsAdapt::test_format_stats_box_hist_defaults`
- `test_stats_enhancements.py::TestIssue1_DefaultFieldsAdapt::test_format_stats_box_profile_defaults`
- `test_stats_enhancements.py::TestIssue1_DefaultFieldsAdapt::test_format_stats_box_scatter_defaults`
- `test_stats_enhancements.py::TestIssue2_2D_n_Semantics::test_2d_n_with_all_valid`
- `test_stats_enhancements.py::TestIssue2_2D_n_Semantics::test_2d_n_with_range_counts_both_valid_in_range`
- `test_stats_enhancements.py::TestIssue2_RangeAwareStats::test_1d_stats_respect_range`
- `test_stats_enhancements.py::TestIssue2_RangeAwareStats::test_2d_stats_respect_range`
- `test_stats_enhancements.py::TestIssue2_RangeAwareStats::test_empty_range_1d`
- `test_stats_enhancements.py::TestIssue2_RangeAwareStats::test_range_boundaries_inclusive`
- `test_stats_enhancements.py::TestIssue2_RangeAwareStats::test_range_with_group_by`
- `test_stats_enhancements.py::TestIssue2_RangeAwareStats::test_range_x_only_for_2d`
- `test_stats_enhancements.py::TestIssue3_RobustStats::test_get_default_stats_fields_robust`
- `test_stats_enhancements.py::TestIssue3_RobustStats::test_get_default_stats_fields_robust_2d_unchanged`
- `test_stats_enhancements.py::TestIssue3_RobustStats::test_mad_computed_when_robust`
- `test_stats_enhancements.py::TestIssue3_RobustStats::test_median_computed_when_robust`
- `test_stats_enhancements.py::TestIssue3_RobustStats::test_quartiles_computed_when_robust`
- `test_stats_enhancements.py::TestIssue3_RobustStats::test_robust_values_correct`
- `test_stats_enhancements.py::TestIssue3_RobustStats::test_robust_with_range`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_all_nan_values`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_correlation_single_point`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_correlation_two_points`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_empty_dataframe`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_expression_evaluation`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_format_stats_box_nan_display`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_nan_values_excluded_1d`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_nan_values_excluded_2d`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_negative_correlation`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_single_point_1d`
- `test_stats_enhancements.py::TestStatsEdgeCases::test_single_point_2d`

</details>

<details><summary><code>test_style.py</code> (4)</summary>

- `test_style.py::TestDefaultStyle::test_default_style_has_required_keys`
- `test_style.py::TestSetStyle::test_set_style_invalid_key_raises`
- `test_style.py::TestSetStyle::test_set_style_invalid_name_raises`
- `test_style.py::TestSetStyle::test_set_style_invalid_type_raises`

</details>

<details><summary><code>test_validation_display.py</code> (3)</summary>

- `test_validation_display.py::TestReferenceOverlay::test_overlay_custom_styling`
- `test_validation_display.py::TestReferenceOverlay::test_overlay_returns_none_for_empty_axis`
- `test_validation_display.py::TestStatisticsBox::test_statistics_box_empty_values`

</details>

<details><summary><code>test_vector.py</code> (16)</summary>

- `test_vector.py::TestParserParenInsideBracket::test_function_on_both_sides`
- `test_vector.py::TestParserParenInsideBracket::test_nested_paren`
- `test_vector.py::TestParserScalar::test_parse_scalar_1d_unchanged`
- `test_vector.py::TestParserScalar::test_parse_scalar_with_computed_y`
- `test_vector.py::TestParserScalar::test_parse_scalar_with_paren_both`
- `test_vector.py::TestParserScalar::test_parse_two_colons_still_raises`
- `test_vector.py::TestParserVectorBroadcast::test_1d_vector`
- `test_vector.py::TestParserVectorBroadcast::test_empty_bracket_raises`
- `test_vector.py::TestParserVectorBroadcast::test_single_element_bracket_is_scalar`
- `test_vector.py::TestParserVectorBroadcast::test_whitespace_in_bracket`
- `test_vector.py::TestVectorChannels::test_group_style_invalid_raises`
- `test_vector.py::TestVectorContract::test_stats_list_entries_are_dicts`
- `test_vector.py::TestVectorContract::test_ylabel_no_common_prefix`
- `test_vector.py::TestVectorDispatch::test_draw_2d_auto_dispatches_scatter`
- `test_vector.py::TestVectorInDrawBatch::test_vector_in_draw_batch`
- `test_vector.py::TestVectorMixedRanges::test_mixed_x_ranges_no_warning`

</details>

### Allow-listed (unclaimed) — 64

Tests classified in `TEST_LAYERS` but not yet feature-claimed.
Each carries a `target_phase` for claim-back tracking.

<details><summary>target_phase: <code>13.27.DF</code> (50)</summary>

- `test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_7_1element_degrades_to_scalar` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_8_empty_list_raises` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_1_facet_by_quartile_with_selection_vector` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_2_facet_by_quartile_with_weights_vector` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_3_facet_inner_compose_no_crash` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_4_facet_by_bins_with_selection_vector` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_5_column_mode_facet_by_with_selection_vector_AD78` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_6_facet_by_quantiles_with_selection_vector_AD79` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_1_profile_vector_path_idempotent` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_2_profile_scalar_path_no_extra_assign` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_3_hist_vector_path_idempotent` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_4_scatter_vector_path_idempotent` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_5_facet_dispatch_idempotent` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_1_per_curve_sanitize_stats_in_stats_list` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_2_nan_policy_filter_default_no_crash` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_3_nan_policy_warn_emits_warning` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_4_no_nan_data_no_sanitize_warning` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_6_single_y_selection_vector_outer` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_7_single_y_selection_vector_inner_raises_actionable` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_8_no_fix1_userwarning_fires` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_9_inner_raise_message_names_lengths` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_WDH_4_single_x_weights_vector_outer` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_1_no_vec_iteration_indices_backward_compat` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_2_profile_no_vector_unchanged` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_3_hist_no_vector_unchanged` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_4_scatter_no_vector_unchanged` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_1_hist_accepts_selection_vector` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_2_hist_selection_vector_bins_shared` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Scatter::test_SDS_1_scatter_accepts_selection_vector` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Scatter::test_SDS_2_scatter_selection_with_facet_by` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_1_selection_weights_explicit_rule` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_2_5channel_refuses_without_facet` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_3_3channel_with_selection_delta_resolves` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_4_explicit_rules_count` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_5_combination_with_quantiles` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_1_hist_weights_vector_runs` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_2_hist_weights_vector_with_global` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_2_2ch_weights_vector_inner` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_5_weights_categorical_kwarg_accepted` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Scatter::test_WDS_2_scatter_weights_vector_silently_dropped` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_groupby_dispatches_correctly` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_invalid_channel_name_raises` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_quantiles_one_subplot_per_quantile` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_vector_creates_n_subplots` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetCapacity::test_facet_max_capacity_fires` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetCapacity::test_facet_max_warn_mode` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_existing_test_facetstar_unchanged` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_facet_true_eqivalent_to_facet_by_groupby` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_old_facet_profile_kwargs_preserved` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed
- `test_phase_13_27_facet_refactor.py::TestFacetSameTrueExclusion::test_facet_by_and_same_true_raises` — seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed

</details>

<details><summary>target_phase: <code>13.28.DF</code> (9)</summary>

- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_combined_autorange_overrides` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_k_robust_override` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_strategy_override` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_k_outlier_in_default_style` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_k_robust_in_default_style` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_percentile_in_default_style` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_strategy_in_default_style` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestNoRegressionInExistingStyleKeys::test_profile_marker_still_present` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed
- `test_phase_13_28_df_fix1_autorange_style_keys.py::TestNoRegressionInExistingStyleKeys::test_quantile_band_alpha_still_present` — seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed

</details>

<details><summary>target_phase: <code>13.32.DF</code> (5)</summary>

- `test_phase_13_32_groupby_quantiles_facet.py::TestRepro::test_in_126_overlay_with_quantiles` — seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed
- `test_phase_13_32_groupby_quantiles_facet.py::TestRepro::test_in_129_facet_with_groupby_bins` — seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed
- `test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_cap_still_fires_without_binning` — seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed
- `test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_with_bins_honors_n` — seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed
- `test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_with_quantiles_honors_n` — seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed

</details>

---

*For per-feature test lists, see `CAPABILITY_MATRIX.html`.*

*Auto-generated. ✅ = invariance (A ≡ B). ☑️ = smoke. 👁 = visual_primitive (orthogonal).*