"""
Test Layer Classification — dfdraw

Determines whether each test is invariance (A ≡ B) or smoke (no crash).
Used by generate_capability_matrix.py to assign ✅ Verified vs ☑️ Smoke-only.

Classification rule:
  - invariance: test checks A == B (two paths produce same result)
  - smoke: test checks code runs without error, or structural properties

Everything not listed defaults to "smoke".

Phase 13.15.DF
"""

TEST_LAYERS = {

    # ── Invariance: batch ≡ standalone (numerical equality) ──
    "test_batch_groups.py::test_value_correctness": "invariance",

    # ── Invariance: same=True identity (ax1 is ax2) ──
    "test_same.py::TestSameBasic::test_same_reuses_axes_profile": "invariance",
    "test_same.py::TestSameBasic::test_same_reuses_axes_hist": "invariance",
    "test_same.py::TestSameBasic::test_same_three_overlays": "invariance",
    "test_same.py::TestSameOverride::test_ax_precedence": "invariance",
    "test_same.py::TestSameAcrossMethods::test_profile_on_hist2d": "invariance",
    "test_same.py::TestSameAcrossMethods::test_profile_on_hexbin": "invariance",
    "test_same.py::TestSameAcrossMethods::test_hist_overlay": "invariance",
    "test_same.py::TestSameAcrossMethods::test_draw_dispatch_same": "invariance",

    # ── Invariance: PyArrow ≡ pandas parity ──
    "test_pyarrow_input.py::TestResultsParity::test_hist_parity": "invariance",
    "test_pyarrow_input.py::TestResultsParity::test_scatter_parity": "invariance",
    "test_pyarrow_input.py::TestResultsParity::test_profile_parity": "invariance",
    "test_pyarrow_input.py::TestResultsParity::test_hist2d_parity": "invariance",
    "test_pyarrow_input.py::TestResultsParity::test_hexbin_parity": "invariance",

    # Everything else defaults to "smoke"
}
