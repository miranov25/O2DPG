"""
Tests for Phase 13.9.ADF: PolynomialSpec + register_function + register_polynomial_from_subframe

Unit tests (§8.1) + Invariance tests (§8.2)
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

# Import PolynomialSpec — adjust path as needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from PolynomialSpec import PolynomialSpec
except ImportError:
    from AliasDataFrame.PolynomialSpec import PolynomialSpec


# =============================================================================
# §8.1 Unit Tests
# =============================================================================

class TestBasisExpressions:
    """Test basis expression generation."""

    def test_degree_332(self):
        """48 terms generated for 3D (3,3,2)."""
        spec = PolynomialSpec(['xM', 'driftM', 'dsecM'], (3, 3, 2))
        assert spec.n_terms == (3+1) * (3+1) * (2+1)  # 48

    def test_4d(self):
        """96 terms for 4D (3,3,2,1)."""
        spec = PolynomialSpec(['xM', 'driftM', 'dsecM', 'tgSlp'], (3, 3, 2, 1))
        assert spec.n_terms == (3+1) * (3+1) * (2+1) * (1+1)  # 96

    def test_2d(self):
        """12 terms for 2D (2,3)."""
        spec = PolynomialSpec(['x', 'y'], (2, 3))
        assert spec.n_terms == 3 * 4  # 12

    def test_sparse(self):
        """term_filter reduces term count."""
        spec_full = PolynomialSpec(['x', 'y', 'z'], (3, 3, 3))
        spec_odd = PolynomialSpec(['x', 'y', 'z'], (3, 3, 3),
                                   term_filter=lambda e: e[2] % 2 == 1)
        assert spec_odd.n_terms < spec_full.n_terms
        # Odd z powers: 1, 3 → 2 values out of 4
        assert spec_odd.n_terms == 4 * 4 * 2  # 32

    def test_key_name_format(self):
        """Key names follow <var><order>_<var><order> convention."""
        spec = PolynomialSpec(['x', 'y'], (1, 1))
        tuples = spec.basis_expressions()
        key_names = [t[0] for t in tuples]
        assert 'x0_y0' in key_names
        assert 'x1_y0' in key_names
        assert 'x0_y1' in key_names
        assert 'x1_y1' in key_names

    def test_expression_format(self):
        """Expressions are valid Python."""
        spec = PolynomialSpec(['x', 'y'], (2, 1))
        tuples = spec.basis_expressions()
        expr_dict = {t[0]: t[1] for t in tuples}
        assert expr_dict['x0_y0'] == '1'
        assert expr_dict['x1_y0'] == 'x'
        assert expr_dict['x2_y0'] == 'x**2'
        assert expr_dict['x1_y1'] == 'x*y'

    def test_constant_term_present(self):
        """Constant term (all zeros exponents) is always first candidate."""
        spec = PolynomialSpec(['a', 'b', 'c'], (2, 2, 2))
        tuples = spec.basis_expressions()
        expressions = [t[1] for t in tuples]
        assert '1' in expressions

    def test_columns_degrees_mismatch_raises(self):
        """Mismatched columns and degrees raises ValueError."""
        with pytest.raises(ValueError):
            PolynomialSpec(['x', 'y'], (1, 2, 3))

    def test_tuple_format(self):
        """basis_expressions returns list of (str, str) tuples."""
        spec = PolynomialSpec(['x'], (2,))
        tuples = spec.basis_expressions()
        assert len(tuples) == 3  # 1, x, x**2
        for t in tuples:
            assert isinstance(t, tuple)
            assert len(t) == 2
            assert isinstance(t[0], str)
            assert isinstance(t[1], str)


class TestSchemaRoundtrip:
    """Test schema serialization."""

    def test_full_roundtrip(self):
        """to_schema → from_schema preserves all fields."""
        spec = PolynomialSpec(['xM', 'driftM', 'dsecM', 'tgSlp'], (3, 3, 2, 1))
        schema = spec.to_schema()
        spec2 = PolynomialSpec.from_schema(schema)
        assert spec2.columns == spec.columns
        assert spec2.degrees == spec.degrees
        assert spec2.n_terms == spec.n_terms
        assert spec2.terms == spec.terms

    def test_sparse_roundtrip(self):
        """Sparse polynomial preserves explicit terms list."""
        spec = PolynomialSpec(['x', 'y'], (3, 3),
                               term_filter=lambda e: e[0] + e[1] <= 3)
        schema = spec.to_schema()
        assert 'terms' in schema  # Sparse → terms present
        spec2 = PolynomialSpec.from_schema(schema)
        assert spec2.n_terms == spec.n_terms
        assert spec2.terms == spec.terms

    def test_terms_omitted_for_full(self):
        """Full polynomial schema has no terms field."""
        spec = PolynomialSpec(['x', 'y'], (2, 3))
        schema = spec.to_schema()
        assert 'terms' not in schema

    def test_terms_present_for_sparse(self):
        """Sparse polynomial schema includes terms."""
        spec = PolynomialSpec(['x', 'y'], (2, 3),
                               term_filter=lambda e: e[0] <= 1)
        schema = spec.to_schema()
        assert 'terms' in schema

    def test_schema_is_json_serializable(self):
        """Schema can be serialized to JSON."""
        import json
        spec = PolynomialSpec(['xM', 'driftM'], (3, 3))
        schema = spec.to_schema()
        json_str = json.dumps(schema)
        schema2 = json.loads(json_str)
        spec2 = PolynomialSpec.from_schema(schema2)
        assert spec2.n_terms == spec.n_terms


class TestRootExpression:
    """Test ROOT C++ expression generation."""

    def test_basic(self):
        """Generates valid expression string."""
        spec = PolynomialSpec(['x', 'y'], (1, 1))
        coeffs = np.array([1.0, 0.5, 0.1, 0.01])
        expr = spec.to_root_expression(coeffs)
        assert 'x' in expr
        assert 'y' in expr
        assert '1.0' in expr or '(1.0' in expr

    def test_zero_coeffs_skipped(self):
        """Zero coefficients are omitted."""
        spec = PolynomialSpec(['x'], (2,))
        coeffs = np.array([1.0, 0.0, 0.5])
        expr = spec.to_root_expression(coeffs)
        # Only 2 terms (constant + x**2), not 3
        # Count ' + ' (with spaces) to avoid matching 'e+00' in scientific notation
        assert expr.count(' + ') <= 1


class TestRegisterFunction:
    """Test register_function on AliasDataFrame."""

    def test_basic(self):
        """Registered function callable from alias."""
        df = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
        adf = AliasDataFrame(df)

        def double(arr):
            return np.asarray(arr) * 2

        adf.register_function('double', double)
        adf.add_alias('y', 'double(x)')
        adf.materialize_alias('y')
        np.testing.assert_array_equal(adf.df['y'].values, [2.0, 4.0, 6.0])

    def test_collision_raises(self):
        """Duplicate name raises ValueError."""
        df = pd.DataFrame({'x': [1.0]})
        adf = AliasDataFrame(df)
        adf.register_function('f', lambda x: x)
        with pytest.raises(ValueError, match="already registered"):
            adf.register_function('f', lambda x: x)

    def test_overwrite(self):
        """overwrite=True allows replacement."""
        df = pd.DataFrame({'x': [1.0]})
        adf = AliasDataFrame(df)
        adf.register_function('f', lambda x: np.asarray(x) * 2)
        adf.register_function('f', lambda x: np.asarray(x) * 3, overwrite=True)
        adf.add_alias('y', 'f(x)')
        adf.materialize_alias('y')
        assert adf.df['y'].values[0] == 3.0

    def test_backward_compatibility(self):
        """Existing aliases unaffected by _default_functions change."""
        df = pd.DataFrame({'x': [1.0, 4.0, 9.0]})
        adf = AliasDataFrame(df)
        adf.add_alias('y', 'sqrt(x)')
        adf.materialize_alias('y')
        np.testing.assert_allclose(adf.df['y'].values, [1.0, 2.0, 3.0])


class TestRegisterPolynomial:
    """Test end-to-end polynomial registration with subframe coefficients."""

    @pytest.fixture
    def setup_adf(self):
        """Create test ADF with 3 sectors, polynomial in 2 variables."""
        df_main = pd.DataFrame({
            'sector': [0, 0, 0, 1, 1, 1, 2, 2, 2],
            'x': np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.float64),
            'y': np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=np.float64),
        })
        # pol(x,y) = c00 + c10*x + c01*y + c11*x*y
        df_coeffs = pd.DataFrame({
            'sector': [0, 1, 2],
            'c_x0_y0': [1.0, 2.0, 3.0],
            'c_x1_y0': [0.1, 0.2, 0.3],
            'c_x0_y1': [0.01, 0.02, 0.03],
            'c_x1_y1': [0.001, 0.002, 0.003],
        })
        adf = AliasDataFrame(df_main)
        adf.register_subframe('Poly', AliasDataFrame(df_coeffs),
                              index_columns=['sector'])
        return adf, df_main, df_coeffs

    def test_end_to_end(self, setup_adf):
        """spec → register → materialize → verify values."""
        adf, df_main, df_coeffs = setup_adf

        spec = PolynomialSpec(['x', 'y'], (1, 1))
        # Coefficient columns must match itertools.product order:
        # (0,0)=1, (0,1)=y, (1,0)=x, (1,1)=x*y
        coeff_cols = ['c_x0_y0', 'c_x0_y1', 'c_x1_y0', 'c_x1_y1']
        adf.register_polynomial_from_subframe('polTest', spec, 'Poly', coeff_cols)
        adf.add_alias('correction', 'polTest(x, y)')
        adf.materialize_alias('correction')

        # Verify analytically
        for i in range(len(df_main)):
            s = df_main.loc[i, 'sector']
            x = df_main.loc[i, 'x']
            y = df_main.loc[i, 'y']
            expected = (df_coeffs.loc[s, 'c_x0_y0']
                       + df_coeffs.loc[s, 'c_x1_y0'] * x
                       + df_coeffs.loc[s, 'c_x0_y1'] * y
                       + df_coeffs.loc[s, 'c_x1_y1'] * x * y)
            assert abs(adf.df.loc[i, 'correction'] - expected) < 1e-10


class TestRepr:
    """Test string representation."""

    def test_full(self):
        spec = PolynomialSpec(['x', 'y'], (2, 3))
        r = repr(spec)
        assert 'PolynomialSpec' in r
        assert '(2, 3)' in r

    def test_sparse(self):
        spec = PolynomialSpec(['x', 'y'], (2, 3),
                               term_filter=lambda e: e[0] + e[1] <= 2)
        r = repr(spec)
        assert 'sparse' in r


# =============================================================================
# §8.2 Invariance Tests
# =============================================================================

class TestInvariancePolynomial:
    """Semantic invariance tests per §8.2."""

    def test_invariance_polynomial_vs_numpy(self):
        """Polynomial alias result == explicit numpy reference."""
        df_main = pd.DataFrame({
            'group': [0, 0, 1, 1],
            'x': np.array([1.0, 2.0, 3.0, 4.0]),
            'y': np.array([0.5, 1.5, 2.5, 3.5]),
        })
        df_coeffs = pd.DataFrame({
            'group': [0, 1],
            'c_x0_y0': [1.0, 2.0],
            'c_x1_y0': [0.1, 0.2],
            'c_x0_y1': [0.01, 0.02],
            'c_x1_y1': [0.001, 0.002],
        })
        adf = AliasDataFrame(df_main)
        adf.register_subframe('C', AliasDataFrame(df_coeffs), index_columns=['group'])

        spec = PolynomialSpec(['x', 'y'], (1, 1))
        # Must match itertools.product order: (0,0)=1, (0,1)=y, (1,0)=x, (1,1)=x*y
        coeff_cols = ['c_x0_y0', 'c_x0_y1', 'c_x1_y0', 'c_x1_y1']
        adf.register_polynomial_from_subframe('pol', spec, 'C', coeff_cols)
        adf.add_alias('result', 'pol(x, y)')
        adf.materialize_alias('result')
        numba_result = adf.df['result'].values

        # Numpy reference
        numpy_result = np.empty(4)
        for i in range(4):
            g = df_main.loc[i, 'group']
            x, y = df_main.loc[i, 'x'], df_main.loc[i, 'y']
            c = df_coeffs[df_coeffs['group'] == g].iloc[0]
            numpy_result[i] = c['c_x0_y0'] + c['c_x1_y0']*x + c['c_x0_y1']*y + c['c_x1_y1']*x*y

        np.testing.assert_allclose(numba_result, numpy_result, atol=1e-10)

    def test_invariance_flat_vs_subframe_coeffs(self):
        """Flat coefficient array == subframe lookup for single-group data."""
        df_main = pd.DataFrame({
            'group': [0, 0, 0, 0],
            'x': np.array([1.0, 2.0, 3.0, 4.0]),
        })
        df_coeffs = pd.DataFrame({
            'group': [0],
            'c_x0': [5.0],
            'c_x1': [0.3],
            'c_x2': [0.01],
        })
        adf = AliasDataFrame(df_main)
        adf.register_subframe('C', AliasDataFrame(df_coeffs), index_columns=['group'])

        spec = PolynomialSpec(['x'], (2,))
        adf.register_polynomial_from_subframe('pol', spec, 'C', ['c_x0', 'c_x1', 'c_x2'])
        adf.add_alias('result', 'pol(x)')
        adf.materialize_alias('result')
        subframe_result = adf.df['result'].values

        # Flat evaluation (no subframe)
        coeffs = np.array([5.0, 0.3, 0.01])
        flat_result = coeffs[0] + coeffs[1] * df_main['x'].values + coeffs[2] * df_main['x'].values**2

        np.testing.assert_allclose(subframe_result, flat_result, atol=1e-10)

    def test_invariance_schema_roundtrip_values(self):
        """Schema roundtrip reconstructs polynomial with identical values."""
        spec = PolynomialSpec(['x', 'y'], (2, 2))
        schema = spec.to_schema()
        spec2 = PolynomialSpec.from_schema(schema)

        # Same basis expressions
        assert spec.basis_expressions() == spec2.basis_expressions()

        # Same ROOT expression for same coefficients
        coeffs = np.random.randn(spec.n_terms)
        assert spec.to_root_expression(coeffs) == spec2.to_root_expression(coeffs)
