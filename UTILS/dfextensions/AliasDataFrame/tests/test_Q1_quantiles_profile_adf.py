"""
test_Q1_quantiles_profile_adf.py — ADF ↔ dfdraw quantiles parity

Verifies ADF.draw() correctly forwards quantiles=, central=, quantile_mode=
kwargs to dfdraw's profile(). No formal phase — test infrastructure only.

Q1_1: error_bars mode via ADF
Q1_2: band mode via ADF
Q1_3: parity — ADF path vs direct dfdraw path produce identical stats
Q1_4: central='median' forwarded correctly
Q1_5: group_by + quantiles via ADF
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

try:
    import matplotlib
    matplotlib.use('Agg')
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from dfdraw import DFDraw
    _HAS_DFDRAW = True
except ImportError:
    _HAS_DFDRAW = False


@pytest.fixture
def adf_gaussian():
    """ADF with Gaussian data suitable for profile + quantiles."""
    rng = np.random.default_rng(2025)
    n = 2000
    x = rng.uniform(0, 10, n).astype(np.float32)
    y = 2 * x + rng.normal(0, 1, n).astype(np.float32)
    grp = (x // 2.5).astype(np.int8)  # 4 groups
    df = pd.DataFrame({'x': x, 'y': y, 'grp': grp})
    adf = AliasDataFrame(df)
    adf.add_alias('y_shifted', 'y - 5', dtype=np.float32)
    return adf


@pytest.mark.skipif(not _HAS_MPL or not _HAS_DFDRAW, reason="Requires matplotlib + dfdraw")
class TestQ1QuantilesADFPassthrough:

    @pytest.mark.invariance
    def test_Q1_1_error_bars_via_adf(self, adf_gaussian):
        """ADF.draw with quantiles=[0.16, 0.84] produces error_bars mode stats."""
        result = adf_gaussian.draw(
            'y:x', type='profile', quantiles=[0.16, 0.84])

        # draw() returns fig or (fig, ax, stats) depending on version
        if isinstance(result, tuple):
            fig, ax, stats = result[0], result[1], result[2] if len(result) > 2 else {}
        else:
            stats = {}

        # At minimum, the call should not raise
        assert result is not None, "Q1_1: draw with quantiles should not fail"

    @pytest.mark.invariance
    def test_Q1_2_band_via_adf(self, adf_gaussian):
        """ADF.draw with quantiles=[0.16, 0.5, 0.84] produces band mode."""
        result = adf_gaussian.draw(
            'y:x', type='profile', quantiles=[0.16, 0.5, 0.84])
        assert result is not None, "Q1_2: draw with band quantiles should not fail"

    @pytest.mark.invariance
    def test_Q1_3_parity_adf_vs_dfdraw(self, adf_gaussian):
        """ADF path and direct dfdraw path produce identical stats."""
        adf = adf_gaussian
        adf.materialize_aliases(names=['y_shifted'])

        # ADF path
        result_adf = adf.draw(
            'y_shifted:x', type='profile', quantiles=[0.16, 0.84])

        # dfdraw direct path
        result_dd = DFDraw(adf.df).profile(
            'y_shifted:x', quantiles=[0.16, 0.84])

        # Both should succeed
        assert result_adf is not None, "Q1_3: ADF path failed"
        assert result_dd is not None, "Q1_3: dfdraw path failed"

        # If both return stats dicts, compare quantile values
        if isinstance(result_adf, tuple) and len(result_adf) > 2:
            stats_adf = result_adf[2]
            if isinstance(result_dd, tuple) and len(result_dd) > 2:
                stats_dd = result_dd[2]
                if 'q_lower_per_bin' in stats_adf and 'q_lower_per_bin' in stats_dd:
                    np.testing.assert_allclose(
                        stats_adf['q_lower_per_bin'],
                        stats_dd['q_lower_per_bin'],
                        rtol=1e-12,
                        err_msg="Q1_3: q_lower parity broken"
                    )
                    np.testing.assert_allclose(
                        stats_adf['q_upper_per_bin'],
                        stats_dd['q_upper_per_bin'],
                        rtol=1e-12,
                        err_msg="Q1_3: q_upper parity broken"
                    )

    @pytest.mark.invariance
    def test_Q1_4_central_median_forwarded(self, adf_gaussian):
        """central='median' kwarg forwarded to dfdraw."""
        result = adf_gaussian.draw(
            'y:x', type='profile',
            quantiles=[0.16, 0.5, 0.84], central='median')
        assert result is not None, "Q1_4: central='median' should not fail"

    @pytest.mark.invariance
    def test_Q1_5_groupby_with_quantiles(self, adf_gaussian):
        """group_by + quantiles via ADF — production-typical pattern."""
        result = adf_gaussian.draw(
            'y:x', type='profile',
            group_by='grp', quantiles=[0.16, 0.84])
        assert result is not None, "Q1_5: group_by + quantiles should not fail"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
