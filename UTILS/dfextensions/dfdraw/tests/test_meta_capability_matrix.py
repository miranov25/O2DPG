"""
Phase 13.49.DF — Meta-tests for the Capability Matrix

Four invariance tests that guard the capability→test traceability the matrix
exposes:

  M.1  test_taxonomy_tests_resolve
       Every test_id claimed by a feature exists in pytest's collected set.
       No grandfathering — must pass on commit. The 4 dangling refs that
       prompted this test were repaired in feature_taxonomy.py as part of
       this phase.

  M.2  test_classification_coverage
       Every test_id in TEST_LAYERS is claimed by some feature OR is in the
       hardcoded KNOWN_UNCLAIMED allow-list with valid required fields. The
       allow-list is bounded, versioned governance (§3.7 of the proposal) —
       not a silent escape hatch. Net growth past the seed baseline warns;
       missing required fields fail hard; visual_primitive tests are never
       allow-listed (asserted by M.4 separately).

  M.3  test_html_emitter_parseable
       Generating the HTML matrix on a synthetic test-results report
       produces parseable HTML with at least len(FEATURES) feature anchors
       and at least one expandable test-list per feature.

  M.4  test_no_orphan_visual_tests
       Every visual_primitive test in TEST_LAYERS is claimed by some
       feature (any category — visual evidence is orthogonal to status,
       §3.5 of the proposal). Visual tests are never allow-listed.

KNOWN_UNCLAIMED is seeded with the 64 TEST_LAYERS-classified invariance
tests that have no feature claim at phase open. Each carries its source
phase (13.27.DF / 13.28.DF / 13.32.DF) so the matrix can surface a claim-
back schedule, not a 64-deep "OPEN" pile.
"""

import os
import re
import subprocess
import sys
import warnings

# Make tests/ importable when running from the repo root
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from feature_taxonomy import FEATURES
from test_layer_classification import TEST_LAYERS


# ---------------------------------------------------------------------------
# KNOWN_UNCLAIMED — hardcoded allow-list, versioned alongside the meta-tests.
# Format (§3.7): each entry is a dict with required keys
#   "test_id"      (str)
#   "reason"       (str, >= 10 chars)
#   "target_phase" (str matching r"^\d+\.\d+\.DF$" or the literal "OPEN")
#
# Discipline:
#   - Missing required fields  -> M.2 fails hard.
#   - Net growth past SEED_BASELINE -> M.2 warns (informational, not a fail;
#     legitimate growth is possible — but every new entry must be reviewed).
#   - Removing an entry whose test is still in TEST_LAYERS and still
#     unclaimed -> M.2 fails naturally (the test re-becomes an unmatched
#     classified entry, which is exactly what M.2 forbids).
#   - visual_primitive tests must NEVER appear here (asserted by M.4).
# ---------------------------------------------------------------------------

KNOWN_UNCLAIMED = [
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_7_1element_degrades_to_scalar", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestComposeInnerOuter::test_CIO_8_empty_list_raises", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_1_facet_by_quartile_with_selection_vector", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_2_facet_by_quartile_with_weights_vector", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_3_facet_inner_compose_no_crash", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_4_facet_by_bins_with_selection_vector", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_5_column_mode_facet_by_with_selection_vector_AD78", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestFacetWithVectorCompose::test_FVC_6_facet_by_quantiles_with_selection_vector_AD79", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_1_profile_vector_path_idempotent", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_2_profile_scalar_path_no_extra_assign", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_3_hist_vector_path_idempotent", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_4_scatter_vector_path_idempotent", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestIdempotency_AllPlotTypes::test_IAP_5_facet_dispatch_idempotent", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_1_per_curve_sanitize_stats_in_stats_list", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_2_nan_policy_filter_default_no_crash", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_3_nan_policy_warn_emits_warning", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestNanPolicyPropagation::test_NPP_4_no_nan_data_no_sanitize_warning", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_6_single_y_selection_vector_outer", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_7_single_y_selection_vector_inner_raises_actionable", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_8_no_fix1_userwarning_fires", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_SDP_9_inner_raise_message_names_lengths", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestPhase_13_27_Commit2_FIX1::test_WDH_4_single_x_weights_vector_outer", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_1_no_vec_iteration_indices_backward_compat", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_2_profile_no_vector_unchanged", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_3_hist_no_vector_unchanged", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestProductionPatternBackwardCompat::test_PPB_4_scatter_no_vector_unchanged", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_1_hist_accepts_selection_vector", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Hist::test_SDH_2_hist_selection_vector_bins_shared", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Scatter::test_SDS_1_scatter_accepts_selection_vector", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionDelta_Scatter::test_SDS_2_scatter_selection_with_facet_by", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_1_selection_weights_explicit_rule", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_2_5channel_refuses_without_facet", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_3_3channel_with_selection_delta_resolves", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_4_explicit_rules_count", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestSelectionWeightsCombined::test_SWC_5_combination_with_quantiles", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_1_hist_weights_vector_runs", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Hist::test_WDH_2_hist_weights_vector_with_global", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_2_2ch_weights_vector_inner", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Profile::test_WDP_5_weights_categorical_kwarg_accepted", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_commit2_selection_weights.py::TestWeightsDelta_Scatter::test_WDS_2_scatter_weights_vector_silently_dropped", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_groupby_dispatches_correctly", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_invalid_channel_name_raises", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_quantiles_one_subplot_per_quantile", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetByChannel::test_facet_by_vector_creates_n_subplots", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetCapacity::test_facet_max_capacity_fires", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetCapacity::test_facet_max_warn_mode", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_existing_test_facetstar_unchanged", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_facet_true_eqivalent_to_facet_by_groupby", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetLegacyEquivalence::test_old_facet_profile_kwargs_preserved", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_27_facet_refactor.py::TestFacetSameTrueExclusion::test_facet_by_and_same_true_raises", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.27.DF not yet feature-claimed", "target_phase": "13.27.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_combined_autorange_overrides", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_k_robust_override", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeKeysAcceptedBySetStyle::test_set_style_accepts_strategy_override", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_k_outlier_in_default_style", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_k_robust_in_default_style", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_percentile_in_default_style", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestAutorangeStyleKeysRegistered::test_autorange_strategy_in_default_style", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestNoRegressionInExistingStyleKeys::test_profile_marker_still_present", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_28_df_fix1_autorange_style_keys.py::TestNoRegressionInExistingStyleKeys::test_quantile_band_alpha_still_present", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.28.DF not yet feature-claimed", "target_phase": "13.28.DF"},
    {"test_id": "test_phase_13_32_groupby_quantiles_facet.py::TestRepro::test_in_126_overlay_with_quantiles", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed", "target_phase": "13.32.DF"},
    {"test_id": "test_phase_13_32_groupby_quantiles_facet.py::TestRepro::test_in_129_facet_with_groupby_bins", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed", "target_phase": "13.32.DF"},
    {"test_id": "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_cap_still_fires_without_binning", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed", "target_phase": "13.32.DF"},
    {"test_id": "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_with_bins_honors_n", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed", "target_phase": "13.32.DF"},
    {"test_id": "test_phase_13_32_groupby_quantiles_facet.py::TestSubfix1::test_facet_groupby_with_quantiles_honors_n", "reason": "seeded at phase 13.49 open - invariance test from Phase 13.32.DF not yet feature-claimed", "target_phase": "13.32.DF"}
]

# Size at v1.0 implementation — used to surface net growth in M.2 output.
SEED_BASELINE = 64

# Allowed target_phase values
_TARGET_PHASE_RE = re.compile(r"^\d+\.\d+\.DF$|^OPEN$")


def _validate_known_unclaimed():
    """Return list of (entry_index, error_string). Empty list = clean."""
    errors = []
    seen_ids = set()
    for i, e in enumerate(KNOWN_UNCLAIMED):
        if not isinstance(e, dict):
            errors.append((i, f"entry is not a dict: {type(e).__name__}"))
            continue
        for k in ("test_id", "reason", "target_phase"):
            if k not in e:
                errors.append((i, f"missing required key '{k}'"))
        if "test_id" in e:
            if not isinstance(e["test_id"], str) or not e["test_id"]:
                errors.append((i, "test_id must be a non-empty str"))
            elif e["test_id"] in seen_ids:
                errors.append((i, f"duplicate test_id: {e['test_id']}"))
            else:
                seen_ids.add(e["test_id"])
        if "reason" in e:
            if not isinstance(e["reason"], str) or len(e["reason"]) < 10:
                errors.append((i, f"reason must be a str of >= 10 chars, got "
                                  f"{type(e['reason']).__name__} len="
                                  f"{len(e['reason']) if isinstance(e['reason'], str) else 'NA'}"))
        if "target_phase" in e:
            if not isinstance(e["target_phase"], str) or not _TARGET_PHASE_RE.match(e["target_phase"]):
                errors.append((i, f"target_phase must match r'^\\d+\\.\\d+\\.DF$|^OPEN$', "
                                  f"got {e['target_phase']!r}"))
    return errors


_KNOWN_UNCLAIMED_IDS = {e["test_id"] for e in KNOWN_UNCLAIMED if isinstance(e, dict) and "test_id" in e}


# ---------------------------------------------------------------------------
# Helper — get pytest's collected test set as repo-relative node IDs
# ---------------------------------------------------------------------------

def _collected_test_ids():
    """Return the set of pytest-collected test node-IDs in basename form.

    Pytest's reported node-IDs depend on the rootdir it discovers (walking up
    looking for conftest.py / pytest.ini / pyproject.toml). When pytest runs
    with O2DPG as the rootdir, node IDs come back as
    `UTILS/dfextensions/dfdraw/tests/test_foo.py::...`; when it runs with
    dfdraw as rootdir, they come as `tests/test_foo.py::...`. `feature_
    taxonomy.py` stores tests as basenames, so this function must reduce to
    the basename regardless of prefix depth (same defect class as the
    matrix generator P0-1 caught by Opus2 + Sonnet53_R2 earlier today)."""
    cwd = os.path.dirname(_HERE)
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=cwd,
    )
    ids = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if "::" not in line:
            continue
        if line.startswith(("ERROR", "WARNING", "====")):
            continue
        if "/" in line:
            line = line.split("/")[-1]
        ids.add(line)
    return ids


# ---------------------------------------------------------------------------
# M.1 — every claimed test_id exists in pytest collection (no grandfathering)
# ---------------------------------------------------------------------------

def test_taxonomy_tests_resolve():
    """M.1: every test_id in any feature's `tests` list exists in pytest's
    collected set. Must pass on commit; no grandfathering."""
    claimed = {t for f in FEATURES for t in f["tests"]}
    collected = _collected_test_ids()
    dangling = sorted(claimed - collected)
    assert not dangling, (
        f"feature_taxonomy.py references {len(dangling)} test IDs that pytest "
        f"cannot collect (typos, renames, or deletions):\n  - "
        + "\n  - ".join(dangling)
    )


# ---------------------------------------------------------------------------
# M.2 — TEST_LAYERS coverage with required-fields allow-list
# ---------------------------------------------------------------------------

def test_classification_coverage():
    """M.2: every TEST_LAYERS test_id is claimed by some feature OR appears in
    KNOWN_UNCLAIMED with all required fields valid. Net growth past
    SEED_BASELINE warns; missing fields fail hard (§3.7)."""
    # First: hard-fail if any allow-list entry is malformed.
    errors = _validate_known_unclaimed()
    assert not errors, (
        f"KNOWN_UNCLAIMED has {len(errors)} malformed entries:\n  - "
        + "\n  - ".join(f"[{i}] {msg}" for i, msg in errors)
    )

    # Surface size + composition for review-time visibility
    size = len(KNOWN_UNCLAIMED)
    print(f"\n  KNOWN_UNCLAIMED size: {size} (seed baseline {SEED_BASELINE})")
    if size > SEED_BASELINE:
        warnings.warn(
            f"KNOWN_UNCLAIMED has grown beyond seed: {size} > "
            f"{SEED_BASELINE}. Every new entry must be reviewed; prefer "
            f"feature-claiming over allow-listing.",
            UserWarning, stacklevel=2,
        )

    # Coverage: every classified test must be claimed-or-allowlisted
    claimed = {t for f in FEATURES for t in f["tests"]}
    orphans = sorted(
        t for t in TEST_LAYERS
        if t not in claimed and t not in _KNOWN_UNCLAIMED_IDS
    )
    assert not orphans, (
        f"{len(orphans)} test(s) classified in TEST_LAYERS are neither "
        f"claimed by a feature nor in KNOWN_UNCLAIMED:\n  - "
        + "\n  - ".join(orphans)
    )


# ---------------------------------------------------------------------------
# M.3 — HTML emitter produces parseable output with >= len(FEATURES) anchors
# ---------------------------------------------------------------------------

def test_html_emitter_parseable():
    """M.3: the HTML emitter, given a synthetic test-results report, returns
    parseable HTML with >= len(FEATURES) feature anchors and at least one
    expandable test-list per feature (§9 D-A: count is `>=`, not `==`)."""
    # Synthesize a minimal test_results report: every claimed test passes,
    # every unclaimed-and-not-allow-listed test is missing (–).
    claimed = sorted({t for f in FEATURES for t in f["tests"]})
    test_results = {t: "passed" for t in claimed}

    # Import the emitter from the matrix generator (built in this phase).
    # Locate scripts/generate_capability_matrix.py relative to this file.
    scripts_dir = os.path.join(os.path.dirname(_HERE), "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import generate_capability_matrix as gen

    html = gen.generate_html_matrix(
        test_results=test_results,
        features=FEATURES,
        test_layers=TEST_LAYERS,
        known_unclaimed=KNOWN_UNCLAIMED,
        phase="13.49.DF",
    )

    # Parseable: must contain the required structural anchors
    assert isinstance(html, str) and html.strip(), "HTML emitter returned empty/non-str"
    assert "<html" in html and "</html>" in html, "HTML missing <html>/</html>"

    # >= len(FEATURES) anchors: each feature row carries an id="feature-{feature_id}"
    n_anchors = sum(html.count(f'id="feature-{f["id"]}"') for f in FEATURES)
    assert n_anchors >= len(FEATURES), (
        f"HTML has {n_anchors} feature anchors, expected >= {len(FEATURES)}"
    )

    # At least one expandable test-list per feature — marker is the test-list
    # container. The emitter uses class="tests-panel" for the expandable block.
    n_panels = html.count('class="tests-panel"')
    assert n_panels >= len(FEATURES), (
        f"HTML has {n_panels} test-panels, expected >= {len(FEATURES)}"
    )


# ---------------------------------------------------------------------------
# M.4 — every visual_primitive test claimed by some feature; never allow-listed
# ---------------------------------------------------------------------------

def test_no_orphan_visual_tests():
    """M.4 (relaxed v1.2 per panel P2-2): every TEST_LAYERS test with layer
    `"visual_primitive"` is claimed by SOME feature (any category — visual
    evidence is orthogonal). Visual tests must NEVER appear in
    KNOWN_UNCLAIMED."""
    claimed = {t for f in FEATURES for t in f["tests"]}
    visual = [t for t, layer in TEST_LAYERS.items() if layer == "visual_primitive"]
    orphans = sorted(t for t in visual if t not in claimed)
    assert not orphans, (
        f"{len(orphans)} visual_primitive test(s) have no feature claim:\n  - "
        + "\n  - ".join(orphans)
    )
    in_allowlist = sorted(t for t in visual if t in _KNOWN_UNCLAIMED_IDS)
    assert not in_allowlist, (
        "visual_primitive tests must never be allow-listed; "
        f"{len(in_allowlist)} found in KNOWN_UNCLAIMED:\n  - "
        + "\n  - ".join(in_allowlist)
    )
