"""
Batch 4 — I17: Full Pipeline Integration Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE. All tests are marked @pytest.mark.invariance.

SCOPE (per v1.2 §4.4)
---------------------
End-to-end integration test: register_subframe + add_alias + set_metadata +
export_tree + read_tree + materialize round-trip produces the same
values as a direct in-memory materialization. Tests that the full
ADF pipeline preserves the values of a simple compound expression
through ROOT serialization.

SCOPE BOUNDARY
--------------
Uses linear subframe join + arithmetic only — no compression (covered
by I11), no dtype pinning (covered by I16), no draw (covered by I7).
Purpose is to catch bugs that only appear when multiple phases are
combined.
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame

try:
    import ROOT
    _HAS_ROOT = ROOT is not None
except ImportError:
    _HAS_ROOT = False

_requires_root = pytest.mark.skipif(
    not _HAS_ROOT, reason="ROOT not available"
)


class TestI17FullPipelineInvariance:

    @_requires_root
    @pytest.mark.invariance
    def test_I17_1_register_export_read_materialize_roundtrip(self, tmp_path):
        """
        I17_1 INVARIANT:
            Build an ADF with a subframe + alias + metadata, materialize
            alias in-memory (reference path), then export_tree +
            read_tree + apply_schema on a fresh instance and materialize
            same alias again. Values must match to float32 tolerance;
            axis title must survive; subframe must be re-registered
            from ROOT (load_subframes=True default).
        CODE PATH:
            Build -> materialize (reference)
            Build -> set_axis_title -> export_tree -> read_tree ->
                     materialize (pipeline)
        """
        n = 150
        rng = np.random.RandomState(17171)
        df_main = pd.DataFrame({
            'sector': rng.randint(0, 10, n).astype(np.int8),
            'pt': rng.uniform(0.5, 5.0, n).astype(np.float32),
        })
        df_sub = pd.DataFrame({
            'sector': np.arange(10, dtype=np.int8),
            'gain': np.linspace(0.9, 1.1, 10).astype(np.float32),
        })

        # Reference: build + materialize in memory
        adf_ref = AliasDataFrame(df_main.copy())
        adf_ref.register_subframe('S', AliasDataFrame(df_sub.copy()),
                                  index_columns='sector')
        adf_ref.add_alias('calib_pt', 'S.gain * pt')
        adf_ref.materialize_aliases(names=['calib_pt'])
        ref_values = adf_ref.df['calib_pt'].values.copy()

        # Pipeline: build + metadata + export + read + materialize
        adf = AliasDataFrame(df_main.copy())
        adf.register_subframe('S', AliasDataFrame(df_sub.copy()),
                              index_columns='sector')
        adf.add_alias('calib_pt', 'S.gain * pt')
        adf.set_axis_title('calib_pt', 'calibrated p_{T} [GeV/c]')

        path = str(tmp_path / 'i17_pipeline.root')
        adf.export_tree(path, treename='tree')

        adf2 = AliasDataFrame.read_tree(path, treename='tree')
        # read_tree should restore subframes and aliases via schema
        adf2.materialize_aliases(names=['calib_pt'])
        pipeline_values = adf2.df['calib_pt'].values

        np.testing.assert_allclose(
            pipeline_values, ref_values,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg=(
                "I17_1: pipeline calib_pt values differ from in-memory "
                "reference"
            )
        )

        # Axis title must survive the roundtrip
        assert adf2.get_axis_title('calib_pt') == 'calibrated p_{T} [GeV/c]', (
            f"I17_1: axis title lost through ROOT roundtrip. "
            f"Got: {adf2.get_axis_title('calib_pt')!r}"
        )
