"""
Tests for Phase 13.13.GB — Expression-Based Linear Columns.

Tests that linear_columns accepts (key_name, expression) tuples
alongside plain column name strings. Covers both make_parallel_fit_v4
and make_sliding_window_fit.
"""
import numpy as np
import pandas as pd
import pytest

try:
    from groupby_regression_optimized import make_parallel_fit_v4
except ImportError:
    from ..groupby_regression_optimized import make_parallel_fit_v4

try:
    from groupby_regression_sliding_window import make_sliding_window_fit
except ImportError:
    from ..groupby_regression_sliding_window import make_sliding_window_fit


# ── Fixtures ──

@pytest.fixture
def sample_df():
    """DataFrame with known polynomial relationship: y = 2*x + 3*x^2 + 1."""
    rng = np.random.RandomState(42)
    n = 2000
    x = rng.uniform(-2, 2, n)
    noise = rng.normal(0, 0.01, n)
    y = 1.0 + 2.0 * x + 3.0 * x ** 2 + noise
    return pd.DataFrame({
        'gBin': np.repeat(np.arange(5), n // 5),
        'x': x,
        'y': y,
    })


@pytest.fixture
def sample_sw_df():
    """DataFrame for sliding window with expression columns."""
    rng = np.random.RandomState(42)
    frames = []
    for xb in range(5):
        for yb in range(5):
            n = 50
            x = rng.standard_normal(n)
            y = 1.0 + 2.0 * x + 0.5 * x ** 2 + rng.normal(0, 0.1, n)
            frames.append(pd.DataFrame({
                'xBin': xb, 'yBin': yb,
                'x': x, 'y': y,
            }))
    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════════
# Test 1: String column unchanged (regression)
# ═══════════════════════════════════════════════════════════════

class TestStringUnchanged:

    def test_string_column_v4(self, sample_df):
        """Plain string linear_columns still works in V4."""
        _, dfGB = make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=['x'], suffix='_v4',
        )
        assert 'y_slope_x_v4' in dfGB.columns

    def test_string_column_sw(self, sample_sw_df):
        """Plain string linear_columns still works in SW."""
        result = make_sliding_window_fit(
            df=sample_sw_df, gb_columns=['xBin', 'yBin'],
            fit_columns=['y'], linear_columns=['x'],
            window_spec={'xBin': 1, 'yBin': 0}, min_stat=5, suffix='_sw',
        )
        assert 'y_slope_x_sw' in result.columns


# ═══════════════════════════════════════════════════════════════
# Test 2: Tuple expression evaluated (smoke)
# ═══════════════════════════════════════════════════════════════

class TestTupleExpression:

    def test_tuple_v4(self, sample_df):
        """V4 accepts (key, expression) tuple."""
        _, dfGB = make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=['x', ('x2', 'x**2')], suffix='_v4',
        )
        assert 'y_slope_x_v4' in dfGB.columns
        assert 'y_slope_x2_v4' in dfGB.columns

    def test_tuple_sw(self, sample_sw_df):
        """SW accepts (key, expression) tuple."""
        result = make_sliding_window_fit(
            df=sample_sw_df, gb_columns=['xBin', 'yBin'],
            fit_columns=['y'],
            linear_columns=['x', ('x2', 'x**2')],
            window_spec={'xBin': 1, 'yBin': 0}, min_stat=5, suffix='_sw',
        )
        assert 'y_slope_x_sw' in result.columns
        assert 'y_slope_x2_sw' in result.columns


# ═══════════════════════════════════════════════════════════════
# Test 3: String not a column raises (error)
# ═══════════════════════════════════════════════════════════════

class TestErrorHandling:

    def test_string_not_column_raises_v4(self, sample_df):
        with pytest.raises((ValueError, KeyError), match="not found|Missing"):
            make_parallel_fit_v4(
                df=sample_df, gb_columns='gBin', fit_columns='y',
                linear_columns=['x', 'NONEXISTENT'], suffix='_v4',
            )

    def test_invalid_key_name_raises(self, sample_df):
        with pytest.raises(ValueError, match="not a valid"):
            make_parallel_fit_v4(
                df=sample_df, gb_columns='gBin', fit_columns='y',
                linear_columns=[('123bad', 'x**2')], suffix='_v4',
            )

    def test_key_collision_raises(self, sample_df):
        """Key name matches existing column → ValueError."""
        with pytest.raises(ValueError, match="conflicts"):
            make_parallel_fit_v4(
                df=sample_df, gb_columns='gBin', fit_columns='y',
                linear_columns=[('x', 'x**2')], suffix='_v4',  # 'x' already exists
            )

    def test_duplicate_key_raises(self, sample_df):
        """Duplicate key_name within tuples → ValueError."""
        with pytest.raises(ValueError, match="Duplicate"):
            make_parallel_fit_v4(
                df=sample_df, gb_columns='gBin', fit_columns='y',
                linear_columns=[('x2', 'x**2'), ('x2', 'x**3')], suffix='_v4',
            )

    def test_bad_expression_raises(self, sample_df):
        """Invalid expression → ValueError with context."""
        with pytest.raises(ValueError, match="Failed to evaluate"):
            make_parallel_fit_v4(
                df=sample_df, gb_columns='gBin', fit_columns='y',
                linear_columns=[('bad', 'NONEXISTENT_COL**2')], suffix='_v4',
            )


# ═══════════════════════════════════════════════════════════════
# Test 6: Naming correct (smoke)
# ═══════════════════════════════════════════════════════════════

class TestNaming:

    def test_output_column_names_v4(self, sample_df):
        """Output columns use key_name, not expression."""
        _, dfGB = make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=[
                'x',
                ('xSq', 'x**2'),
                ('xCu', 'x**3'),
            ], suffix='_v4',
        )
        assert 'y_slope_x_v4' in dfGB.columns
        assert 'y_slope_xSq_v4' in dfGB.columns
        assert 'y_slope_xCu_v4' in dfGB.columns
        # No raw expression in column names
        for col in dfGB.columns:
            assert '**' not in col, f"Raw expression in column name: {col}"


# ═══════════════════════════════════════════════════════════════
# Test 7: Mixed string + tuple (smoke)
# ═══════════════════════════════════════════════════════════════

class TestMixed:

    def test_mixed_v4(self, sample_df):
        """Mix of plain strings and tuples works."""
        _, dfGB = make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=['x', ('x2', 'x**2')], suffix='_v4',
        )
        # Both slopes should be present and finite
        assert np.isfinite(dfGB['y_slope_x_v4']).all()
        assert np.isfinite(dfGB['y_slope_x2_v4']).all()


# ═══════════════════════════════════════════════════════════════
# Test 8: Expression matches materialized (INVARIANCE)
# ═══════════════════════════════════════════════════════════════

class TestInvariance:

    def test_expression_matches_materialized_v4(self, sample_df):
        """Expression column ≡ pre-materialized column (V4)."""
        # Materialized
        df_mat = sample_df.copy()
        df_mat['x2'] = df_mat['x'] ** 2
        _, dfGB_mat = make_parallel_fit_v4(
            df=df_mat, gb_columns='gBin', fit_columns='y',
            linear_columns=['x', 'x2'], suffix='_v4',
        )

        # Expression
        _, dfGB_expr = make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=['x', ('x2', 'x**2')], suffix='_v4',
        )

        for col in ['y_slope_x_v4', 'y_slope_x2_v4', 'y_intercept_v4']:
            np.testing.assert_array_equal(
                dfGB_mat[col].values, dfGB_expr[col].values,
                err_msg=f"Expression ≠ materialized: {col}")

    def test_expression_matches_materialized_sw(self, sample_sw_df):
        """Expression column ≡ pre-materialized column (SW)."""
        # Materialized
        df_mat = sample_sw_df.copy()
        df_mat['x2'] = df_mat['x'] ** 2
        result_mat = make_sliding_window_fit(
            df=df_mat, gb_columns=['xBin', 'yBin'],
            fit_columns=['y'], linear_columns=['x', 'x2'],
            window_spec={'xBin': 1, 'yBin': 0}, min_stat=5, suffix='_sw',
        )

        # Expression
        result_expr = make_sliding_window_fit(
            df=sample_sw_df, gb_columns=['xBin', 'yBin'],
            fit_columns=['y'],
            linear_columns=['x', ('x2', 'x**2')],
            window_spec={'xBin': 1, 'yBin': 0}, min_stat=5, suffix='_sw',
        )

        keys = ['xBin', 'yBin']
        mat_s = result_mat.sort_values(keys).reset_index(drop=True)
        expr_s = result_expr.sort_values(keys).reset_index(drop=True)

        for col in ['y_slope_x_sw', 'y_slope_x2_sw', 'y_intercept_sw']:
            np.testing.assert_array_equal(
                mat_s[col].values, expr_s[col].values,
                err_msg=f"Expression ≠ materialized: {col}")


# ═══════════════════════════════════════════════════════════════
# Test: Original DataFrame not mutated (P1-6)
# ═══════════════════════════════════════════════════════════════

class TestNoMutation:

    def test_original_df_not_mutated_v4(self, sample_df):
        """Original DataFrame columns unchanged after expression fit."""
        cols_before = set(sample_df.columns)
        make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=['x', ('x2', 'x**2')], suffix='_v4',
        )
        assert set(sample_df.columns) == cols_before, \
            "Original DataFrame was mutated by expression columns"

    def test_original_df_not_mutated_sw(self, sample_sw_df):
        """Original DataFrame columns unchanged after expression fit (SW)."""
        cols_before = set(sample_sw_df.columns)
        make_sliding_window_fit(
            df=sample_sw_df, gb_columns=['xBin', 'yBin'],
            fit_columns=['y'],
            linear_columns=['x', ('x2', 'x**2')],
            window_spec={'xBin': 1, 'yBin': 0}, min_stat=5, suffix='_sw',
        )
        assert set(sample_sw_df.columns) == cols_before, \
            "Original DataFrame was mutated by expression columns"


# ═══════════════════════════════════════════════════════════════
# Test: Metadata contains linear_column_map
# ═══════════════════════════════════════════════════════════════

class TestMetadata:

    def test_metadata_v4(self, sample_df):
        """V4 metadata includes linear_column_map when expressions used."""
        _, _, metadata = make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=['x', ('x2', 'x**2')], suffix='_v4',
            return_metadata=True,
        )
        assert 'linear_column_map' in metadata
        assert metadata['linear_column_map']['x'] == 'x'
        assert metadata['linear_column_map']['x2'] == 'x**2'
        assert metadata['linear_columns_normalized'] == ['x', 'x2']

    def test_metadata_sw(self, sample_sw_df):
        """SW metadata includes linear_column_map when expressions used."""
        result, metadata = make_sliding_window_fit(
            df=sample_sw_df, gb_columns=['xBin', 'yBin'],
            fit_columns=['y'],
            linear_columns=['x', ('x2', 'x**2')],
            window_spec={'xBin': 1, 'yBin': 0}, min_stat=5, suffix='_sw',
            return_metadata=True,
        )
        assert 'linear_column_map' in metadata
        assert metadata['linear_column_map']['x2'] == 'x**2'

    def test_no_metadata_for_string_only(self, sample_df):
        """No linear_column_map in metadata when all string columns."""
        _, _, metadata = make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=['x'], suffix='_v4',
            return_metadata=True,
        )
        assert 'linear_column_map' not in metadata


# ═══════════════════════════════════════════════════════════════
# Test: Polynomial workflow (integration)
# ═══════════════════════════════════════════════════════════════

class TestPolynomialWorkflow:

    def test_polynomial_fit_recovers_coefficients(self, sample_df):
        """3-term polynomial fit recovers known coefficients."""
        # y = 1.0 + 2.0*x + 3.0*x^2
        _, dfGB = make_parallel_fit_v4(
            df=sample_df, gb_columns='gBin', fit_columns='y',
            linear_columns=['x', ('x2', 'x**2')], suffix='_v4',
        )
        # Check coefficients (averaged across groups)
        mean_intercept = dfGB['y_intercept_v4'].mean()
        mean_slope_x = dfGB['y_slope_x_v4'].mean()
        mean_slope_x2 = dfGB['y_slope_x2_v4'].mean()

        np.testing.assert_allclose(mean_intercept, 1.0, atol=0.05,
                                   err_msg="Intercept not recovered")
        np.testing.assert_allclose(mean_slope_x, 2.0, atol=0.1,
                                   err_msg="x slope not recovered")
        np.testing.assert_allclose(mean_slope_x2, 3.0, atol=0.1,
                                   err_msg="x^2 slope not recovered")
