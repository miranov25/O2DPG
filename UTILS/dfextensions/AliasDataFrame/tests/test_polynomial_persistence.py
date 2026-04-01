"""
Tests for Phase 13.9.Fix1: Polynomial function persistence

Registered polynomial functions must survive schema export/import cycle.
After loading, polynomial aliases should be functional without manual
re-registration.
"""

import pytest
import numpy as np
import pandas as pd
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame
from PolynomialSpec import PolynomialSpec


@pytest.fixture
def adf_with_polynomial():
    """ADF with registered polynomial from subframe."""
    df_main = pd.DataFrame({
        'group': np.array([0, 0, 1, 1, 0, 1]),
        'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        'y': np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]),
    })
    # Coefficients: group 0 → [1.0, 0.5, 0.2, 0.05], group 1 → [2.0, -0.3, 0.1, -0.02]
    # Terms: x^0*y^0, x^1*y^0, x^0*y^1, x^1*y^1
    df_coeffs = pd.DataFrame({
        'group': [0, 1],
        'c_x0_y0': [1.0, 2.0],
        'c_x1_y0': [0.5, -0.3],
        'c_x0_y1': [0.2, 0.1],
        'c_x1_y1': [0.05, -0.02],
    })

    adf = AliasDataFrame(df_main)
    adf.register_subframe('Coeffs', AliasDataFrame(df_coeffs), index_columns=['group'])

    spec = PolynomialSpec(columns=['x', 'y'], degrees=(1, 1))
    coeff_cols = ['c_x0_y0', 'c_x1_y0', 'c_x0_y1', 'c_x1_y1']
    adf.register_polynomial_from_subframe('poly', spec, 'Coeffs', coeff_cols)
    adf.add_alias('correction', 'poly(x, y)')

    return adf, spec, coeff_cols


class TestPolynomialPersistence:
    """Phase 13.9.Fix1: Polynomial function persistence."""

    def test_schema_contains_registered_functions(self, adf_with_polynomial):
        """export_schema includes registered_functions."""
        adf, _, _ = adf_with_polynomial
        schema = adf.export_schema()
        assert 'registered_functions' in schema
        assert 'poly' in schema['registered_functions']
        assert schema['registered_functions']['poly']['coefficients_subframe'] == 'Coeffs'

    def test_schema_has_polynomial_spec(self, adf_with_polynomial):
        """Schema stores PolynomialSpec details for reconstruction."""
        adf, _, coeff_cols = adf_with_polynomial
        schema = adf.export_schema()
        poly_schema = schema['registered_functions']['poly']
        assert poly_schema['coeff_select'] == coeff_cols
        assert 'columns' in poly_schema or 'dimensions' in poly_schema

    def test_schema_json_serializable(self, adf_with_polynomial):
        """Schema with registered_functions is JSON serializable."""
        adf, _, _ = adf_with_polynomial
        schema = adf.export_schema()
        json_str = json.dumps(schema)
        restored = json.loads(json_str)
        assert 'registered_functions' in restored

    def test_save_load_schema_roundtrip(self, adf_with_polynomial, tmp_path):
        """Schema saves and loads with registered_functions intact."""
        adf, _, _ = adf_with_polynomial
        schema_path = str(tmp_path / 'schema.json')
        adf.save_schema(schema_path)

        loaded_schema = AliasDataFrame.load_schema(schema_path)
        assert 'registered_functions' in loaded_schema
        assert 'poly' in loaded_schema['registered_functions']

    def test_reconstruct_polynomial_from_schema(self, adf_with_polynomial):
        """apply_schema reconstructs polynomial functions."""
        adf, spec, coeff_cols = adf_with_polynomial

        # Get reference values
        adf.materialize_alias('correction')
        reference = adf.df['correction'].values.copy()

        # Export schema
        schema = adf.export_schema()

        # Create new ADF with same data + subframe
        df_main = adf.df[['group', 'x', 'y']].copy()
        sf = adf.get_subframe('Coeffs')
        adf2 = AliasDataFrame(df_main)
        adf2.register_subframe('Coeffs', AliasDataFrame(sf.df.copy()), index_columns=['group'])

        # Apply schema — should reconstruct polynomial
        adf2.apply_schema(schema)

        # Alias should work now
        adf2.materialize_alias('correction')
        np.testing.assert_allclose(adf2.df['correction'].values, reference, atol=1e-10)

    def test_reconstruct_skips_evaluator(self, adf_with_polynomial):
        """Evaluator functions are skipped during reconstruction (by design)."""
        adf, _, _ = adf_with_polynomial

        # Add a fake evaluator entry to schema
        adf._schema['registered_functions']['my_eval'] = {
            'type': 'evaluator',
            'coord_columns': ['x'],
            'predictor_columns': None,
        }

        schema = adf.export_schema()

        # Create new ADF
        df_main = adf.df[['group', 'x', 'y']].copy()
        sf = adf.get_subframe('Coeffs')
        adf2 = AliasDataFrame(df_main)
        adf2.register_subframe('Coeffs', AliasDataFrame(sf.df.copy()), index_columns=['group'])
        adf2.apply_schema(schema)

        # Polynomial should be reconstructed, evaluator should not
        assert 'poly' in adf2._registered_functions
        assert 'my_eval' not in adf2._registered_functions

    def test_reconstruct_missing_subframe_skips(self):
        """Reconstruction skips if subframe not registered."""
        df = pd.DataFrame({'x': [1.0, 2.0], 'y': [0.5, 1.5]})
        adf = AliasDataFrame(df)

        schema = {
            'registered_functions': {
                'poly': {
                    'columns': ['x', 'y'],
                    'degrees': [1, 1],
                    'coefficients_subframe': 'MissingSF',
                    'coeff_select': ['c0', 'c1', 'c2', 'c3'],
                }
            }
        }

        # Should not raise — just skip with message
        adf.apply_schema(schema)
        assert 'poly' not in getattr(adf, '_registered_functions', {})

    @pytest.mark.invariance
    def test_invariance_before_after_schema_roundtrip(self, adf_with_polynomial, tmp_path):
        """Invariance: polynomial values identical before and after schema roundtrip."""
        adf, _, _ = adf_with_polynomial

        # Before
        adf.materialize_alias('correction')
        before = adf.df['correction'].values.copy()

        # Save schema
        schema_path = str(tmp_path / 'schema.json')
        adf.save_schema(schema_path)

        # Reconstruct
        df_main = adf.df[['group', 'x', 'y']].copy()
        sf = adf.get_subframe('Coeffs')
        adf2 = AliasDataFrame(df_main)
        adf2.register_subframe('Coeffs', AliasDataFrame(sf.df.copy()), index_columns=['group'])
        loaded = AliasDataFrame.load_schema(schema_path)
        adf2.apply_schema(loaded)

        # After
        adf2.materialize_alias('correction')
        after = adf2.df['correction'].values.copy()

        np.testing.assert_allclose(before, after, atol=1e-10,
                                    err_msg="Polynomial values differ after schema roundtrip")
