"""
Phase 13.18.ADF — R1: Regression Metadata Persistence Invariance

STANDALONE NEW TEST FILE (§5.1 precedent, Phase 13.12).
Marked @pytest.mark.invariance.

Covers:
    R1_1: register_regression_metadata then export_tree -> read_tree
          preserves the metadata dict EXACTLY (incl. description,
          annotations).

R1_2 (evaluator equivalence after roundtrip) lives in
test_R1_2_evaluator_roundtrip_invariance.py for Capability Matrix
unambiguity (v1.1 P1-B).
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


def _build_adf_with_regression_metadata(suffix='_sw'):
    df_main = pd.DataFrame({
        'sector': np.arange(30, dtype=np.int32) % 3,
        'meanIDC': np.linspace(0.1, 1.0, 30).astype(np.float32),
    })
    n_bins = 3
    df_sub = pd.DataFrame({
        'sector': np.arange(n_bins, dtype=np.int32),
        f'dX_intercept{suffix}': np.linspace(1.0, 3.0, n_bins),
        f'dX_slope_meanIDC{suffix}': np.linspace(0.1, 0.3, n_bins),
        f'dY_intercept{suffix}': np.linspace(2.0, 4.0, n_bins),
        f'dY_slope_meanIDC{suffix}': np.linspace(0.2, 0.4, n_bins),
    })
    adf = AliasDataFrame(df_main)
    adf.register_subframe('TPC_corr', AliasDataFrame(df_sub),
                          index_columns='sector')
    adf.register_regression_metadata(
        'TPC_model',
        subframe_name='TPC_corr',
        group_columns=['sector'],
        predictor_columns=['meanIDC'],
        targets=['dX', 'dY'],
        suffix=suffix,
        fit_intercept=True,
        default_method='lookup',
        default_bounds='nan',
        description='Phase 13.18 R1 regression test',
        annotations={
            'axis_titles': {'dX': 'Delta x [cm]', 'dY': 'Delta y [cm]'},
            'units': {'meanIDC': 'fA/cm'},
        },
    )
    return adf


class TestR1RegressionMetadataPersistence:

    @pytest.mark.invariance
    def test_R1_1_metadata_dict_survives_schema_roundtrip(self, tmp_path):
        """
        R1_1 INVARIANT:
            register_regression_metadata then export_tree/read_tree
            preserves the metadata dict exactly, including description
            and annotations sub-dict.
        PRODUCTION ENTRY POINT:
            adf.register_regression_metadata(...)
            adf.export_tree(path, treename='tree')
            AliasDataFrame.read_tree(path, treename='tree')
        CODE PATH:
            Serialize via _serialize_schema regression_metadata hook
            (patch 2); deserialize via _deserialize_schema hook
            (patch 3); apply_schema restore (patch 4).
        """
        adf = _build_adf_with_regression_metadata()
        meta_before = dict(
            adf._schema['regression_metadata']['TPC_model']
        )

        path = str(tmp_path / 'r1_1.root')
        adf.export_tree(path, treename='tree')

        adf2 = AliasDataFrame.read_tree(path, treename='tree')
        meta_after = adf2._schema.get('regression_metadata', {}).get(
            'TPC_model'
        )

        assert meta_after is not None, (
            "R1_1: 'TPC_model' metadata missing after read_tree roundtrip"
        )

        # Compare field-by-field. _bound_evaluator is transient; ignore.
        compared_fields = [
            'subframe_name', 'group_columns', 'predictor_columns',
            'targets', 'suffix', 'fit_intercept', 'default_method',
            'default_bounds', 'description',
        ]
        for key in compared_fields:
            assert meta_before[key] == meta_after.get(key), (
                f"R1_1: field {key!r} differs. "
                f"before={meta_before[key]!r}, after={meta_after.get(key)!r}"
            )

        # annotations must match as a dict
        assert meta_before['annotations'] == meta_after.get('annotations'), (
            f"R1_1: annotations dict differs. "
            f"before={meta_before['annotations']!r}, "
            f"after={meta_after.get('annotations')!r}"
        )
