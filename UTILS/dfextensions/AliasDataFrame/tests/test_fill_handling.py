"""
test_fill_handling.py - Tests for subframe fill configuration

Tests Phase 1 fill handling implementation:
- set_global_fill() / clear_global_fill()
- set_subframe_fill() / clear_subframe_fill()
- fill_mode='safe' vs 'direct'
- fill_missing, fill_nan, fill_inf, fill_invalid
- Aggregated warning behavior
- Backward compatibility

Test requirements from review:
- test_set_subframe_fill_stores_config
- test_set_subframe_fill_unknown_subframe_raises
- test_fill_missing_replaces_nan
- test_direct_mode_fills_missing
- test_fill_applied_during_materialization
- test_subframe_config_overrides_global
- test_warn_missing_keys_false_suppresses
- test_default_fill_mode_is_safe
- test_subframe_fill_mode_overrides_global
- test_set_subframe_fill_rejects_unknown_fill_mode
- test_safe_mode_does_not_touch_non_subframe_aliases
- test_direct_and_safe_give_same_values_when_no_missing
- test_fill_applied_in_dependency_chain
- test_update_config_affects_subsequent_materialization
- test_auto_generated_aliases_respect_fill_config
- test_fill_accepts_any_scalar_and_rejects_containers (was test_fill_missing_rejects_non_numeric; renamed in round 7, AD-14)
- test_fast_mode_raises_not_implemented
- test_backward_compatibility_no_config
"""

import numpy as np
import pandas as pd
import pytest
import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def main_df():
    """Main DataFrame with some keys that won't exist in subframe."""
    return pd.DataFrame({
        'idx': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    })


@pytest.fixture
def sub_df():
    """Subframe with missing keys (no 3, 7) and some NaN/Inf values."""
    return pd.DataFrame({
        'idx': [1, 2, 4, 5, 6, 8, 9, 10],
        'value': [10.0, 20.0, 40.0, np.nan, np.inf, 80.0, -np.inf, 100.0],
        'clean': [10.0, 20.0, 40.0, 50.0, 60.0, 80.0, 90.0, 100.0],
    })


@pytest.fixture
def adf_with_subframe(main_df, sub_df):
    """AliasDataFrame with registered subframe."""
    adf = AliasDataFrame(main_df)
    sub_adf = AliasDataFrame(sub_df)
    adf.register_subframe('T', sub_adf, index_columns='idx')
    return adf


# =============================================================================
# Test: set_global_fill / clear_global_fill
# =============================================================================

class TestGlobalFillConfig:
    """Tests for global fill configuration."""
    
    def test_default_fill_mode_is_safe(self, adf_with_subframe):
        """Test that default fill_mode is 'safe'."""
        adf = adf_with_subframe
        config = adf._get_fill_config('T')
        assert config['fill_mode'] == 'safe'
    
    def test_set_global_fill_defaults(self, adf_with_subframe):
        """Test that defaults are sensible."""
        adf = adf_with_subframe
        config = adf._get_fill_config('T')
        
        assert config['fill_missing'] is None
        assert config['fill_nan'] is None
        assert config['fill_inf'] is None
        assert config['warn_missing_keys'] is True
        assert config['warn_threshold'] == 0.01
        assert config['fill_mode'] == 'safe'
    
    def test_set_global_fill_missing(self, adf_with_subframe):
        """Test setting global fill_missing."""
        adf = adf_with_subframe
        adf.set_global_fill(fill_missing=0.0)
        
        config = adf._get_fill_config('T')
        assert config['fill_missing'] == 0.0
    
    def test_set_global_fill_invalid_expands(self, adf_with_subframe):
        """Test that fill_invalid sets both fill_nan and fill_inf."""
        adf = adf_with_subframe
        adf.set_global_fill(fill_invalid=-999.0)
        
        config = adf._get_fill_config('T')
        assert config['fill_nan'] == -999.0
        assert config['fill_inf'] == -999.0
    
    def test_set_global_fill_specific_overrides_invalid(self, adf_with_subframe):
        """Test that fill_nan overrides fill_invalid."""
        adf = adf_with_subframe
        adf.set_global_fill(fill_invalid=-999.0, fill_nan=-1.0)
        
        config = adf._get_fill_config('T')
        assert config['fill_nan'] == -1.0  # Specific wins
        assert config['fill_inf'] == -999.0  # From fill_invalid
    
    def test_set_global_fill_mode_validation(self, adf_with_subframe):
        """Test that invalid fill_mode raises error."""
        adf = adf_with_subframe
        with pytest.raises(ValueError, match="fill_mode must be"):
            adf.set_global_fill(fill_mode='invalid')
    
    def test_fast_mode_raises_not_implemented(self, adf_with_subframe):
        """Test that fill_mode='fast' raises NotImplementedError."""
        adf = adf_with_subframe
        with pytest.raises(NotImplementedError, match="Phase 2"):
            adf.set_global_fill(fill_mode='fast')
    
    def test_fill_accepts_any_scalar_and_rejects_containers(self, adf_with_subframe):
        """SUPERSEDED CONTRACT — AD-14/13.76.ADF (architect, 2026-07-28,
        GPT31 Decision 3), rewritten deliberately and flagged in the CRR.

        This test used to assert that ANY non-numeric fill raises
        `TypeError: must be numeric`. The architect overturned that rule:

            "set_subframe_fill() accepts any fill value compatible with the
             actual column dtype: numeric, string, timestamp/NaT, complex, or
             an existing category. It never silently adds a category. It
             raises clearly on incompatibility."

        Compatibility is not decidable at CONFIGURATION time, because a fill
        is configured per SUBFRAME while dtypes are per COLUMN. So the check
        moved to projection (`_coerce_fill_to_dtype`), where the column's own
        dtype is known and the error can name it. What stays refusable here is
        what is wrong for every dtype: a CONTAINER is not a fill value.

        The old assertion is preserved in inverted form below, so the change
        of contract is visible in the test rather than only in a document.
        """
        adf = adf_with_subframe
        # was TypeError — a string is now a legal fill for a string/object
        # column and is refused at projection for, say, a float one.
        adf.set_global_fill(fill_missing="string")
        adf.clear_global_fill()
        # containers were rejected before and still are, with a clearer reason
        with pytest.raises(TypeError, match="scalar fill value"):
            adf.set_global_fill(fill_nan=[1, 2, 3])
        with pytest.raises(TypeError, match="scalar fill value"):
            adf.set_global_fill(fill_missing=np.array([1, 2]))
    
    def test_clear_global_fill(self, adf_with_subframe):
        """Test clearing global fill resets to defaults."""
        adf = adf_with_subframe
        adf.set_global_fill(fill_missing=0.0, fill_mode='direct')
        adf.clear_global_fill()
        
        config = adf._get_fill_config('T')
        assert config['fill_missing'] is None
        assert config['fill_mode'] == 'safe'


# =============================================================================
# Test: set_subframe_fill / clear_subframe_fill
# =============================================================================

class TestSubframeFillConfig:
    """Tests for subframe-specific fill configuration."""
    
    def test_set_subframe_fill_stores_config(self, adf_with_subframe):
        """Test that set_subframe_fill stores configuration."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, warn_missing_keys=False)
        
        assert 'T' in adf._subframe_fill_config
        assert adf._subframe_fill_config['T']['fill_missing'] == 0.0
        assert adf._subframe_fill_config['T']['warn_missing_keys'] is False
    
    def test_set_subframe_fill_unknown_subframe_raises(self, adf_with_subframe):
        """Test that unknown subframe raises error."""
        adf = adf_with_subframe
        with pytest.raises(ValueError, match="not registered"):
            adf.set_subframe_fill('NonExistent', fill_missing=0.0)
    
    def test_set_subframe_fill_rejects_unknown_fill_mode(self, adf_with_subframe):
        """Test that invalid fill_mode raises error."""
        adf = adf_with_subframe
        with pytest.raises(ValueError, match="fill_mode must be"):
            adf.set_subframe_fill('T', fill_mode='invalid')
    
    def test_subframe_config_overrides_global(self, adf_with_subframe):
        """Test that subframe config overrides global."""
        adf = adf_with_subframe
        adf.set_global_fill(fill_missing=0.0)
        adf.set_subframe_fill('T', fill_missing=-1.0)
        
        config = adf._get_fill_config('T')
        assert config['fill_missing'] == -1.0  # Subframe wins
    
    def test_subframe_fill_mode_overrides_global(self, adf_with_subframe):
        """Test that subframe fill_mode overrides global."""
        adf = adf_with_subframe
        adf.set_global_fill(fill_mode='safe')
        adf.set_subframe_fill('T', fill_mode='direct')
        
        config = adf._get_fill_config('T')
        assert config['fill_mode'] == 'direct'
    
    def test_set_subframe_fill_partial_override(self, adf_with_subframe):
        """Test partial override - some from global, some from subframe."""
        adf = adf_with_subframe
        adf.set_global_fill(fill_missing=0.0, fill_nan=-999.0)
        adf.set_subframe_fill('T', fill_missing=-1.0)  # Only override fill_missing
        
        config = adf._get_fill_config('T')
        assert config['fill_missing'] == -1.0  # Subframe
        assert config['fill_nan'] == -999.0    # Global (not overridden)
    
    def test_clear_subframe_fill(self, adf_with_subframe):
        """Test clearing subframe fill reverts to global."""
        adf = adf_with_subframe
        adf.set_global_fill(fill_missing=0.0)
        adf.set_subframe_fill('T', fill_missing=-1.0)
        adf.clear_subframe_fill('T')
        
        config = adf._get_fill_config('T')
        assert config['fill_missing'] == 0.0  # Back to global
    
    def test_clear_subframe_fill_validates_subframe(self, adf_with_subframe):
        """Test that clearing unknown subframe raises error."""
        adf = adf_with_subframe
        with pytest.raises(ValueError, match="not registered"):
            adf.clear_subframe_fill('NonExistent')


# =============================================================================
# Test: fill_mode='direct'
# =============================================================================

class TestFillModeDirect:
    """Tests for fill_mode='direct' - fastest mode."""
    
    def test_direct_mode_fills_missing(self, adf_with_subframe):
        """Test that direct mode fills missing keys."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='direct')
        adf.add_alias('t_clean', 'T.clean')
        adf.materialize_alias('t_clean')
        
        # Keys 3 and 7 are missing in subframe
        assert adf.df['t_clean'].iloc[2] == 0.0  # idx=3
        assert adf.df['t_clean'].iloc[6] == 0.0  # idx=7
        
        # Other values should be correct
        assert adf.df['t_clean'].iloc[0] == 10.0  # idx=1
    
    def test_direct_mode_no_fill(self, adf_with_subframe):
        """Test direct mode with no fill_missing keeps NaN."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_mode='direct')  # No fill_missing
        adf.add_alias('t_clean', 'T.clean')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.materialize_alias('t_clean')
        
        # Missing keys should be NaN
        assert np.isnan(adf.df['t_clean'].iloc[2])  # idx=3
        assert np.isnan(adf.df['t_clean'].iloc[6])  # idx=7
    
    def test_V3_2_unmatched_key_gets_declared_neutral_in_masked_correction(self):
        """Production-shaped extension: direct fill + conditional correction.

        The missing key is a target/TPC row, so this proves that explicit
        fill_missing=0.0 supplies the additive neutral.  Non-target rows carry
        poison calibration values and must remain exactly unchanged by the mask.
        """
        row = np.array([10, 100, 151, 152, 160, 170, 190], dtype=np.int32)
        sec = np.arange(7, dtype=np.int32)
        dyC1 = np.array([10., 20., 30., 40., 50., 60., 70.], dtype=np.float64)
        calib = np.array([1., 2., 3., 999., 999., 999., 999.], dtype=np.float64)
        keep = np.array([0, 2, 3, 4, 5, 6], dtype=np.int64)  # missing sec=1, target row 100

        adf = AliasDataFrame(pd.DataFrame({"row": row, "sec": sec, "dyC1": dyC1}))
        sub = AliasDataFrame(pd.DataFrame({"sec": sec[keep], "calib": calib[keep]}))
        adf.register_subframe("Cal", sub, index_columns=["sec"])
        adf.set_subframe_fill(
            "Cal", fill_missing=0.0, fill_mode="direct", warn_missing_keys=False
        )
        adf.add_alias("dyC2", "dyC1-(Cal.calib*(row<152))")

        got = np.asarray(adf.eval("dyC2"))
        assert got.dtype == dyC1.dtype
        np.testing.assert_allclose(got[0], dyC1[0] - calib[0])
        np.testing.assert_array_equal(got[1], dyC1[1])          # unmatched target -> +0
        np.testing.assert_allclose(got[2], dyC1[2] - calib[2])
        np.testing.assert_array_equal(got[3:], dyC1[3:])        # outside mask

    def test_direct_mode_does_not_fill_original_nan(self, adf_with_subframe):
        """Test direct mode doesn't fill NaN that was in original subframe data."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='direct')
        adf.add_alias('t_value', 'T.value')
        adf.materialize_alias('t_value')
        
        # Missing keys filled with 0.0
        assert adf.df['t_value'].iloc[2] == 0.0  # idx=3 (missing)
        
        # Original NaN in data is NOT filled in direct mode
        # (this is the trade-off for speed)
        assert np.isnan(adf.df['t_value'].iloc[4])  # idx=5 has NaN in subframe


# =============================================================================
# Test: fill_mode='safe'
# =============================================================================

class TestFillModeSafe:
    """Tests for fill_mode='safe' - default, separate handling."""
    
    def test_fill_missing_replaces_nan(self, adf_with_subframe):
        """Test that fill_missing replaces NaN for missing keys."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='safe')
        adf.add_alias('t_clean', 'T.clean')
        adf.materialize_alias('t_clean')
        
        # Missing keys should be filled
        assert adf.df['t_clean'].iloc[2] == 0.0  # idx=3
        assert adf.df['t_clean'].iloc[6] == 0.0  # idx=7
    
    def test_safe_mode_fills_missing_only(self, adf_with_subframe):
        """Test safe mode with fill_missing only."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='safe')
        adf.add_alias('t_value', 'T.value')
        adf.materialize_alias('t_value')
        
        # Missing keys filled
        assert adf.df['t_value'].iloc[2] == 0.0  # idx=3
        assert adf.df['t_value'].iloc[6] == 0.0  # idx=7
        
        # Original NaN NOT filled (fill_nan not set)
        assert np.isnan(adf.df['t_value'].iloc[4])  # idx=5
    
    def test_safe_mode_fills_nan(self, adf_with_subframe):
        """Test safe mode fills NaN in original data."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_nan=-1.0, fill_mode='safe')
        adf.add_alias('t_value', 'T.value')
        adf.materialize_alias('t_value')
        
        # Missing keys filled with fill_missing
        assert adf.df['t_value'].iloc[2] == 0.0  # idx=3
        
        # Original NaN filled with fill_nan
        assert adf.df['t_value'].iloc[4] == -1.0  # idx=5
    
    def test_safe_mode_fills_inf(self, adf_with_subframe):
        """Test safe mode fills Inf values."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_inf=-999.0, fill_mode='safe')
        adf.add_alias('t_value', 'T.value')
        adf.materialize_alias('t_value')
        
        # Inf values filled
        assert adf.df['t_value'].iloc[5] == -999.0  # idx=6 has +Inf
        assert adf.df['t_value'].iloc[8] == -999.0  # idx=9 has -Inf
    
    def test_safe_mode_fills_all_invalid(self, adf_with_subframe):
        """Test safe mode with fill_invalid fills both NaN and Inf."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_invalid=-999.0, fill_mode='safe')
        adf.add_alias('t_value', 'T.value')
        adf.materialize_alias('t_value')
        
        # Missing filled with fill_missing
        assert adf.df['t_value'].iloc[2] == 0.0  # idx=3
        
        # NaN and Inf filled with fill_invalid
        assert adf.df['t_value'].iloc[4] == -999.0  # idx=5 (NaN)
        assert adf.df['t_value'].iloc[5] == -999.0  # idx=6 (+Inf)
        assert adf.df['t_value'].iloc[8] == -999.0  # idx=9 (-Inf)
    
    def test_safe_mode_does_not_touch_non_subframe_aliases(self, main_df):
        """Test that fill config doesn't affect non-subframe aliases."""
        adf = AliasDataFrame(main_df)
        
        # Add a regular alias (no subframe)
        adf.add_alias('x_squared', 'x ** 2')
        adf.materialize_alias('x_squared')
        
        # Values should be computed normally
        assert adf.df['x_squared'].iloc[0] == 1.0  # 1^2
        assert adf.df['x_squared'].iloc[1] == 4.0  # 2^2
        assert adf.df['x_squared'].iloc[2] == 9.0  # 3^2


# =============================================================================
# Test: Mode Comparison
# =============================================================================

class TestModeComparison:
    """Tests comparing 'safe' and 'direct' modes."""
    
    def test_direct_and_safe_give_same_values_when_no_missing(self, main_df):
        """Test that both modes give same results when no missing keys."""
        # Subframe with all keys present
        full_df = pd.DataFrame({
            'idx': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        })
        
        # Test with 'safe' mode
        adf1 = AliasDataFrame(main_df.copy())
        adf1.register_subframe('T', AliasDataFrame(full_df.copy()), index_columns='idx')
        adf1.set_subframe_fill('T', fill_mode='safe')
        adf1.add_alias('t_val', 'T.value')
        adf1.materialize_alias('t_val')
        
        # Test with 'direct' mode
        adf2 = AliasDataFrame(main_df.copy())
        adf2.register_subframe('T', AliasDataFrame(full_df.copy()), index_columns='idx')
        adf2.set_subframe_fill('T', fill_mode='direct')
        adf2.add_alias('t_val', 'T.value')
        adf2.materialize_alias('t_val')
        
        # Results should be identical
        np.testing.assert_array_equal(adf1.df['t_val'].values, adf2.df['t_val'].values)


# =============================================================================
# Test: Warning Behavior
# =============================================================================

class TestWarningBehavior:
    """Tests for warning aggregation and thresholds."""
    
    def test_warning_emitted_above_threshold(self, adf_with_subframe):
        """Test that warning is emitted when missing > threshold."""
        adf = adf_with_subframe
        # 2 of 10 missing = 20% > default 1% threshold
        adf.set_subframe_fill('T', warn_threshold=0.01)
        adf.add_alias('t_clean', 'T.clean')
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf.materialize_aliases(names=['t_clean'])
            
            # Should have warning about missing keys
            assert len(w) >= 1
            assert any("Missing key summary" in str(warning.message) for warning in w)
    
    def test_no_warning_below_threshold(self, adf_with_subframe):
        """Test no warning when missing < threshold."""
        adf = adf_with_subframe
        # 2 of 10 missing = 20%, threshold = 50%
        adf.set_subframe_fill('T', warn_threshold=0.50)
        adf.add_alias('t_clean', 'T.clean')
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf.materialize_aliases(names=['t_clean'])
            
            # Should NOT have missing key warning
            missing_warnings = [x for x in w if "Missing key summary" in str(x.message)]
            assert len(missing_warnings) == 0
    
    def test_warn_missing_keys_false_suppresses(self, adf_with_subframe):
        """Test that warn_missing_keys=False suppresses warnings."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', warn_missing_keys=False)
        adf.add_alias('t_clean', 'T.clean')
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf.materialize_aliases(names=['t_clean'])
            
            missing_warnings = [x for x in w if "Missing key summary" in str(x.message)]
            assert len(missing_warnings) == 0
    
    def test_warning_shows_fill_value(self, adf_with_subframe):
        """Test that warning message includes fill value."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, warn_threshold=0.01)
        adf.add_alias('t_clean', 'T.clean')
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf.materialize_aliases(names=['t_clean'])
            
            warning_msgs = [str(x.message) for x in w]
            combined = ' '.join(warning_msgs)
            assert "filled with 0.0" in combined or "filled with 0" in combined


# =============================================================================
# Test: Materialization Behavior
# =============================================================================

class TestMaterializationBehavior:
    """Tests for fill behavior during materialization."""
    
    def test_fill_applied_during_materialization(self, adf_with_subframe):
        """Test that fill is applied during materialize_aliases."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='direct')
        adf.add_alias('t_clean', 'T.clean')
        
        # Before materialization, column doesn't exist
        assert 't_clean' not in adf.df.columns
        
        adf.materialize_aliases(names=['t_clean'])
        
        # After materialization, column exists with filled values
        assert 't_clean' in adf.df.columns
        assert adf.df['t_clean'].iloc[2] == 0.0  # idx=3 was missing
    
    def test_update_config_affects_subsequent_materialization(self, main_df, sub_df):
        """Test that updating config affects subsequent materializations."""
        # First materialization with fill=0.0
        adf = AliasDataFrame(main_df.copy())
        adf.register_subframe('T', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='direct')
        adf.add_alias('t_clean', 'T.clean')
        adf.materialize_alias('t_clean')
        
        val_first = adf.df['t_clean'].iloc[2]  # idx=3
        assert val_first == 0.0
        
        # Update config and add new alias
        adf.set_subframe_fill('T', fill_missing=-999.0)
        adf.add_alias('t_value', 'T.value')
        adf.materialize_alias('t_value')
        
        val_second = adf.df['t_value'].iloc[2]  # idx=3
        assert val_second == -999.0
    
    def test_fill_applied_in_dependency_chain(self, adf_with_subframe):
        """Test that fill works when alias depends on subframe alias."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='direct')
        
        # Alias that references subframe
        adf.add_alias('t_val', 'T.clean')
        # Alias that depends on first alias
        adf.add_alias('t_val_doubled', 't_val * 2')
        
        adf.materialize_aliases(names=['t_val_doubled'])
        
        # idx=3 was missing, so t_val=0.0, t_val_doubled=0.0
        assert adf.df['t_val_doubled'].iloc[2] == 0.0


# =============================================================================
# Test: Complex Expressions
# =============================================================================

class TestComplexExpressions:
    """Tests that fill config applies to complex expressions."""
    
    def test_fill_applies_to_complex_expression(self, adf_with_subframe):
        """Test fill works with expressions like T.x * T.y + 1."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='direct')
        adf.add_alias('computed', 'T.clean * 2 + 1')
        adf.materialize_alias('computed')
        
        # For idx=3 (missing): 0.0 * 2 + 1 = 1.0
        assert adf.df['computed'].iloc[2] == 1.0
        
        # For idx=1: 10.0 * 2 + 1 = 21.0
        assert adf.df['computed'].iloc[0] == 21.0
    
    @pytest.mark.xfail(reason="Blocked by auto_alias_subframe cycle bug (BUG-2025-11-27-001)")
    def test_auto_generated_aliases_respect_fill_config(self, adf_with_subframe):
        """Test fill works with auto_alias_subframe() generated aliases."""
        adf = adf_with_subframe
        adf.set_subframe_fill('T', fill_missing=0.0, fill_mode='direct')
        result = adf.auto_alias_subframe('T')
        
        # Should have created aliases
        assert len(result['created']) > 0
        
        # Materialize one of them
        if 'clean' in adf.aliases:
            adf.materialize_aliases(names=['clean'])
            # Missing keys should be filled
            assert adf.df['clean'].iloc[2] == 0.0  # idx=3


# =============================================================================
# Test: Multiple Subframes
# =============================================================================

class TestMultipleSubframes:
    """Tests with multiple subframes having different configs."""
    
    def test_different_fill_per_subframe(self, main_df):
        """Test different fill config per subframe."""
        # Create two subframes with different data
        sub1_df = pd.DataFrame({
            'idx': [1, 2, 4, 5],
            'val1': [10.0, 20.0, 40.0, 50.0],
        })
        sub2_df = pd.DataFrame({
            'idx': [1, 3, 5, 7],
            'val2': [100.0, 300.0, 500.0, 700.0],
        })
        
        adf = AliasDataFrame(main_df)
        adf.register_subframe('S1', AliasDataFrame(sub1_df), index_columns='idx')
        adf.register_subframe('S2', AliasDataFrame(sub2_df), index_columns='idx')
        
        # Different fill for each
        adf.set_subframe_fill('S1', fill_missing=-1.0, fill_mode='direct')
        adf.set_subframe_fill('S2', fill_missing=-2.0, fill_mode='direct')
        
        adf.add_alias('v1', 'S1.val1')
        adf.add_alias('v2', 'S2.val2')
        adf.materialize_aliases(names=['v1', 'v2'])
        
        # S1: idx 3,6,7,8,9,10 missing -> -1.0
        assert adf.df['v1'].iloc[2] == -1.0  # idx=3
        
        # S2: idx 2,4,6,8,9,10 missing -> -2.0
        assert adf.df['v2'].iloc[1] == -2.0  # idx=2


# =============================================================================
# Test: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """Tests that existing code works without fill config."""
    
    def test_backward_compatibility_no_config(self, adf_with_subframe):
        """Test that default (no fill config) produces NaN for missing."""
        adf = adf_with_subframe
        adf.add_alias('t_clean', 'T.clean')
        
        # Don't set any fill config
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.materialize_alias('t_clean')
        
        # Missing keys should be NaN (original behavior)
        assert np.isnan(adf.df['t_clean'].iloc[2])  # idx=3
        assert np.isnan(adf.df['t_clean'].iloc[6])  # idx=7


# =============================================================================
# Test: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_subframe(self, main_df):
        """Test handling of empty subframe."""
        empty_df = pd.DataFrame({'idx': pd.Series([], dtype=int), 'value': pd.Series([], dtype=float)})
        
        adf = AliasDataFrame(main_df)
        adf.register_subframe('Empty', AliasDataFrame(empty_df), index_columns='idx')
        adf.set_subframe_fill('Empty', fill_missing=0.0, fill_mode='direct')
        adf.add_alias('e_val', 'Empty.value')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf.materialize_alias('e_val')
        
        # All values should be fill_missing
        assert (adf.df['e_val'] == 0.0).all()
    
    def test_all_keys_present(self, main_df):
        """Test when all main frame keys exist in subframe."""
        full_df = pd.DataFrame({
            'idx': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'value': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        })
        
        adf = AliasDataFrame(main_df)
        adf.register_subframe('Full', AliasDataFrame(full_df), index_columns='idx')
        adf.set_subframe_fill('Full', fill_missing=-999.0, warn_threshold=0.0)
        adf.add_alias('f_val', 'Full.value')
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf.materialize_aliases(names=['f_val'])
            
            # No missing key warning expected (0% missing)
            missing_warnings = [x for x in w if "Missing key summary" in str(x.message)]
            assert len(missing_warnings) == 0
        
        # No fill values should be present
        assert -999.0 not in adf.df['f_val'].values
    
    def test_fill_value_zero_vs_none(self, main_df, sub_df):
        """Test that fill_missing=0.0 is different from fill_missing=None."""
        # With fill_missing=0.0
        adf1 = AliasDataFrame(main_df.copy())
        adf1.register_subframe('T', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf1.set_subframe_fill('T', fill_missing=0.0, fill_mode='direct')
        adf1.add_alias('t_clean', 'T.clean')
        adf1.materialize_alias('t_clean')
        assert adf1.df['t_clean'].iloc[2] == 0.0
        
        # With fill_missing=None
        adf2 = AliasDataFrame(main_df.copy())
        adf2.register_subframe('T', AliasDataFrame(sub_df.copy()), index_columns='idx')
        adf2.set_subframe_fill('T', fill_missing=None, fill_mode='direct')
        adf2.add_alias('t_clean', 'T.clean')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adf2.materialize_alias('t_clean')
        assert np.isnan(adf2.df['t_clean'].iloc[2])


# =============================================================================
# Calibration Workflow Example Test
# =============================================================================

class TestCalibrationWorkflow:
    """Test the intended calibration workflow usage."""
    
    def test_calibration_workflow_example(self, main_df, sub_df):
        """
        Test the calibration workflow as documented in the proposal.
        
        This is the intended usage pattern for calibration:
        - fill_missing=0.0 (missing = no correction)
        - fill_invalid=0.0 (invalid = no correction)
        - warn_missing_keys=False (expected in calibration)
        - fill_mode='direct' (maximum speed)
        """
        adf = AliasDataFrame(main_df)
        sub_adf = AliasDataFrame(sub_df)
        adf.register_subframe('DITS0FitSide', sub_adf, index_columns='idx')
        
        # Configure for calibration workflow
        adf.set_subframe_fill(
            'DITS0FitSide',
            fill_missing=0.0,
            fill_invalid=0.0,
            warn_missing_keys=False,
            fill_mode='direct',
        )
        
        # Create calibration alias
        adf.add_alias('dyC2', 'DITS0FitSide.clean')
        
        # Materialize (should be silent, no warnings)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf.materialize_aliases(names=['dyC2'])
            
            # No warnings expected
            missing_warnings = [x for x in w if "Missing key summary" in str(x.message)]
            assert len(missing_warnings) == 0
        
        # Values should be filled appropriately
        assert adf.df['dyC2'].iloc[2] == 0.0  # idx=3 was missing


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
