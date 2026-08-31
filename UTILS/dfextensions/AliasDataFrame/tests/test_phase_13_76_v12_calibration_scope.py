"""PHASE_13_76_ADF B3.2b — production-shaped calibration scope tests.

These tests are derived from the real makeSmoothMapsWithTPC calibration pattern:
  * detector-scope masking (row < 152),
  * explicit additive-neutral fill_missing=0.0 for unmatched join keys,
  * coefficient/subframe re-registration without redefining the correction alias.

The detector boundary is a production-shaped fixture.  The ADF contract under test is
conditional correction scope + declared neutral fill + interactive recalibration.
"""

import numpy as np
import pandas as pd
import pytest

try:
    from AliasDataFrame.AliasDataFrame import AliasDataFrame
except (ImportError, ModuleNotFoundError):
    from AliasDataFrame import AliasDataFrame


DYN_P0_2 = (
    "DYN-P0-2: re-registering a subframe does not invalidate aliases sourced "
    "from that subframe"
)

ROW = np.array([10, 100, 151, 152, 160, 170, 190], dtype=np.int32)
SEC = np.arange(7, dtype=np.int32)
DYC1 = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)
CALIB_V0 = np.array([1.0, 2.0, 3.0, 999.0, 999.0, 999.0, 999.0], dtype=np.float64)
CALIB_V1 = CALIB_V0 * 3.0
GAPPED_KEEP = np.array([0, 2, 3, 4, 5, 6], dtype=np.int64)


def _main_frame():
    return AliasDataFrame(pd.DataFrame({"row": ROW, "sec": SEC, "dyC1": DYC1}))


def _calibration(values, keep=None):
    if keep is None:
        keep = np.arange(len(SEC), dtype=np.int64)
    return AliasDataFrame(pd.DataFrame({
        "sec": SEC[keep],
        "calib": np.asarray(values)[keep],
    }))


def _register_calibration(parent, values, keep=None):
    parent.register_subframe("Cal", _calibration(values, keep), index_columns=["sec"])
    parent.set_subframe_fill(
        "Cal",
        fill_missing=0.0,
        fill_mode="direct",
        warn_missing_keys=False,
    )


def _fixture_full():
    parent = _main_frame()
    _register_calibration(parent, CALIB_V0)
    parent.add_alias("dyC2", "dyC1-(Cal.calib*(row<152))")
    return parent


def _fixture_gapped():
    parent = _main_frame()
    _register_calibration(parent, CALIB_V0, GAPPED_KEEP)
    parent.add_alias("dyC2", "dyC1-(Cal.calib*(row<152))")
    return parent

@pytest.mark.invariance
class TestV12CalibrationScope:
    def test_v3_1_tpc_scope_corrected_non_tpc_untouched(self):
        parent = _fixture_full()
        got = np.asarray(parent.eval("dyC2"))
        assert got.dtype == DYC1.dtype

        np.testing.assert_allclose(got[:3], DYC1[:3] - CALIB_V0[:3])
        # Exact numerical equality outside the mask.  Poison coefficients make
        # any accidental correction leakage loud.
        np.testing.assert_array_equal(got[3:], DYC1[3:])

@pytest.mark.invariance
class TestV12CalibrationRecalibration:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P0_2)
    def test_v3_3_recalibration_updates_target_rows_only(self):
        parent = _fixture_full()
        parent.materialize_aliases(names=["dyC2"])

        # Deliberately do NOT redefine dyC2 and do NOT manually drop it.
        parent.register_subframe(
            "Cal",
            _calibration(CALIB_V1),
            index_columns=["sec"],
        )

        got = np.asarray(parent.eval("dyC2"))
        assert got.dtype == DYC1.dtype
        np.testing.assert_allclose(got[:3], DYC1[:3] - CALIB_V1[:3])
        np.testing.assert_array_equal(got[3:], DYC1[3:])

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P0_2)
    def test_v3_4_mask_fill_and_recalibration_compose(self):
        parent = _fixture_gapped()
        parent.materialize_aliases(names=["dyC2"])

        # Preserve the same missing-key pattern at the new iteration.
        parent.register_subframe(
            "Cal",
            _calibration(CALIB_V1, GAPPED_KEEP),
            index_columns=["sec"],
        )

        got = np.asarray(parent.eval("dyC2"))
        assert got.dtype == DYC1.dtype
        np.testing.assert_allclose(got[0], DYC1[0] - CALIB_V1[0])
        np.testing.assert_array_equal(got[1], DYC1[1])       # unmatched TPC -> +0
        np.testing.assert_allclose(got[2], DYC1[2] - CALIB_V1[2])
        np.testing.assert_array_equal(got[3], DYC1[3])       # row 152 -> outside mask
        np.testing.assert_array_equal(got[4:], DYC1[4:])
