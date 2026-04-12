"""
Batch 3 — I12: Metadata Persistence Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE (§5.1 deviation note, established in Batch 1).
All tests are marked @pytest.mark.invariance.

SCOPE (per v1.2 §4.3)
---------------------
I12 covers metadata persistence through the schema export/import cycle
(JSON path) and the bulk-vs-individual metadata API equivalence.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
All tests call public API methods directly:
  - set_axis_title(column, title)            @ AliasDataFrame.py:9724
  - set_column_metadata(column, **metadata)  @ AliasDataFrame.py:8193
  - set_columns_metadata(metadata)           @ AliasDataFrame.py:8227
  - export_schema() / apply_schema()         @ AliasDataFrame.py:8111/8645

NOTE: I12_1 verifies schema-level storage. In-memory roundtrip through
apply_schema() is tested, not ROOT-file roundtrip (ROOT path is covered
by I5_2 in Batch 1). This avoids double-testing the ROOT serialization.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf_for_metadata():
    """Small ADF with multiple columns for metadata tests."""
    df = pd.DataFrame({
        'x': np.linspace(-250.0, 250.0, 50).astype(np.float32),
        'y': np.linspace(-250.0, 250.0, 50).astype(np.float32),
        'pt': np.linspace(0.5, 5.0, 50).astype(np.float32),
    })
    return AliasDataFrame(df)


class TestI12MetadataPersistenceInvariance:
    """Phase 13.12 I12 — Metadata persistence through schema roundtrip."""

    @pytest.mark.invariance
    def test_I12_1_axis_title_survives_schema_roundtrip(
        self, adf_for_metadata
    ):
        """
        I12_1 INVARIANT:
            set_axis_title('x', 'x [cm]') followed by
            schema = export_schema() then apply_schema(schema) on a
            fresh ADF preserves the axis title under the 'title' key
            at _schema['columns']['x']['title'] (source line 9740).
        CODE PATH:
            adf.set_axis_title('x', 'x [cm]')          @ line 9724
            schema = adf.export_schema()                @ line 8111
            adf2 = AliasDataFrame(df)
            adf2.apply_schema(schema, warn_missing=False)  @ line 8645
            adf2.get_axis_title('x')                    @ line 9742
        PRODUCTION ENTRY POINT:
            Round-trip workflow for dfdraw axis labels — titles set in
            one session must survive schema export and re-application
            in another session.
        REGRESSION GUARD: axis title persistence through JSON schema.
        """
        adf = adf_for_metadata
        adf.set_axis_title('x', 'x [cm]')
        adf.set_axis_title('pt', 'p_{T} [GeV/c]')

        # Direct schema-level check (source line 9740)
        assert adf._schema['columns']['x']['title'] == 'x [cm]', (
            "I12_1 precondition: set_axis_title did not populate "
            "_schema['columns']['x']['title']"
        )

        schema = adf.export_schema()

        # Fresh ADF on same DataFrame, apply schema
        adf2 = AliasDataFrame(adf.df.copy())
        adf2.apply_schema(schema, warn_missing=False)

        # Verify both titles survived via public getter
        assert adf2.get_axis_title('x') == 'x [cm]', (
            f"I12_1: 'x' axis title lost through schema roundtrip. "
            f"Got: {adf2.get_axis_title('x')!r}"
        )
        assert adf2.get_axis_title('pt') == 'p_{T} [GeV/c]', (
            f"I12_1: 'pt' axis title lost through schema roundtrip. "
            f"Got: {adf2.get_axis_title('pt')!r}"
        )

        # Also verify at schema level (independent of getter implementation)
        assert adf2._schema['columns']['x'].get('title') == 'x [cm]', (
            "I12_1: 'x' title lost at schema level after apply_schema"
        )

    @pytest.mark.invariance
    def test_I12_2_bulk_metadata_equals_individual_metadata_calls(
        self, adf_for_metadata
    ):
        """
        I12_2 INVARIANT:
            set_columns_metadata({col: {**kwargs}, ...}) produces the
            same _schema state as calling set_column_metadata(col, **kwargs)
            once per column.
        CODE PATH:
            Path A: adf_a.set_columns_metadata({x: {..}, y: {..}})   @ 8227
            Path B: adf_b.set_column_metadata('x', **{..})            @ 8193
                    adf_b.set_column_metadata('y', **{..})            @ 8193
            Compare _schema['columns'][col] dicts — must be equal.
        PRODUCTION ENTRY POINT:
            Bulk and individual metadata setters are both public API;
            users expect them to be interchangeable.
        REGRESSION GUARD: set_columns_metadata must not diverge from
            N repeated set_column_metadata calls.
        """
        metadata_dict = {
            'x': {
                'unit': 'cm',
                'axisLabel': 'x [cm]',
                'description': 'TPC cluster x position',
            },
            'y': {
                'unit': 'cm',
                'axisLabel': 'y [cm]',
            },
            'pt': {
                'unit': 'GeV/c',
                'axisLabel': 'p_{T} [GeV/c]',
                'description': 'transverse momentum',
            },
        }

        # Path A: bulk setter
        adf_a = adf_for_metadata
        adf_a.set_columns_metadata(metadata_dict)

        # Path B: fresh ADF with individual setter calls
        df_b = pd.DataFrame({
            'x': adf_a.df['x'].values.copy(),
            'y': adf_a.df['y'].values.copy(),
            'pt': adf_a.df['pt'].values.copy(),
        })
        adf_b = AliasDataFrame(df_b)
        for col, kwargs in metadata_dict.items():
            adf_b.set_column_metadata(col, **kwargs)

        # Compare schema state for each column
        for col in metadata_dict:
            schema_a = adf_a._schema['columns'].get(col, {})
            schema_b = adf_b._schema['columns'].get(col, {})

            # Check every metadata key set via either path agrees
            for key, expected in metadata_dict[col].items():
                assert schema_a.get(key) == expected, (
                    f"I12_2: Path A (bulk) missing or wrong value for "
                    f"{col}.{key}: got {schema_a.get(key)!r}, "
                    f"expected {expected!r}"
                )
                assert schema_b.get(key) == expected, (
                    f"I12_2: Path B (individual) missing or wrong value "
                    f"for {col}.{key}: got {schema_b.get(key)!r}, "
                    f"expected {expected!r}"
                )
                assert schema_a.get(key) == schema_b.get(key), (
                    f"I12_2: bulk/individual divergence for {col}.{key}: "
                    f"bulk={schema_a.get(key)!r}, "
                    f"individual={schema_b.get(key)!r}"
                )
