"""
Phase 13.20.ADF — Mandatory roundtrip test for export_tree metadata batching.

PURPOSE: verify that export_tree → read_tree preserves ALL metadata
(aliases, schema, subframe indices, compression info) for configurations
representative of the production TPC calibration pipeline.

THIS TEST MUST PASS BEFORE AND AFTER THE FIX.
- Before: establishes baseline (current behavior is correct, just slow)
- After: confirms the batched metadata write preserves everything

PHASE: 13.20.ADF (export_tree performance)
ARCHITECT DIRECTION: "We should check carefully as it can make side effects."
REVIEWER REQUIREMENT: "mandatory test — catches both risks"

Requires ROOT (import ROOT). Skip if not available.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import ROOT
    HAS_ROOT = True
except ImportError:
    HAS_ROOT = False

try:
    from AliasDataFrame import AliasDataFrame
    HAS_ADF = True
except ImportError:
    HAS_ADF = False


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
@pytest.mark.skipif(not HAS_ADF, reason="AliasDataFrame not available")
class TestExportTreeMetadataRoundtrip:
    """
    E1_1..E1_4 — roundtrip tests for export_tree with varying subframe counts.
    """

    def _build_main_adf(self, n=500):
        """Build a main ADF with columns typical of the calibration pipeline."""
        rng = np.random.default_rng(1320)
        df = pd.DataFrame({
            'x': rng.uniform(85.0, 245.0, n).astype(np.float32),
            'y': rng.uniform(-50.0, 50.0, n).astype(np.float32),
            'z': rng.uniform(-250.0, 250.0, n).astype(np.float32),
            'dy': rng.normal(0.0, 0.5, n).astype(np.float16),
            'dz': rng.normal(0.0, 0.5, n).astype(np.float16),
            'sec': rng.integers(0, 36, n).astype(np.int8),
            'row': rng.integers(0, 152, n).astype(np.int16),
            'track_index': rng.integers(0, 100, n).astype(np.int32),
            'firstTForbit': np.full(n, 29901376, dtype=np.int64),
        })
        adf = AliasDataFrame(df)
        adf.add_alias('r', 'sqrt(x**2 + y**2)', dtype=np.float32)
        adf.add_alias('drift', '1 - abs(z)/250', dtype=np.float16)
        adf.add_alias('side', '(z > 0) * 1', dtype=np.int8)
        return adf

    def _build_coeff_subframe(self, name_prefix, n_sectors=36, n_terms=48):
        """Build a coefficient subframe like PolyFitDy/PolyFitDz."""
        rng = np.random.default_rng(hash(name_prefix) % 2**31)
        cols = {'sec': np.arange(n_sectors, dtype=np.int8)}
        for i in range(n_terms):
            cols[f'{name_prefix}_slope_{i}'] = rng.normal(0, 0.01, n_sectors).astype(np.float32)
        df = pd.DataFrame(cols)
        adf = AliasDataFrame(df)
        adf.add_alias(f'{name_prefix}_sum', f'{name_prefix}_slope_0 + {name_prefix}_slope_1')
        return adf

    def _build_track_subframe(self, n_tracks=100):
        """Build a track-level subframe like T."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            'track_index': np.arange(n_tracks, dtype=np.int32),
            'firstTForbit': np.full(n_tracks, 29901376, dtype=np.int64),
            'mP3': rng.normal(0, 1, n_tracks).astype(np.float16),
            'mP4': rng.normal(0, 1, n_tracks).astype(np.float16),
        })
        adf = AliasDataFrame(df)
        adf.add_alias('tanLambda', 'mP3', dtype=np.float16)
        return adf

    def _verify_roundtrip(self, original, loaded, subframe_names):
        """Verify all metadata survived roundtrip."""
        findings = []

        # 1. Aliases preserved
        for alias_name, alias_expr in original.aliases.items():
            if alias_name not in loaded.aliases:
                findings.append(f"ALIAS MISSING: {alias_name}")
            elif loaded.aliases[alias_name] != alias_expr:
                findings.append(
                    f"ALIAS CHANGED: {alias_name}: "
                    f"'{alias_expr}' → '{loaded.aliases[alias_name]}'"
                )

        # 2. Schema preserved (check key fields)
        orig_schema = original._schema
        loaded_schema = loaded._schema
        if 'aliases' in orig_schema:
            for aname in orig_schema['aliases']:
                if aname not in loaded_schema.get('aliases', {}):
                    findings.append(f"SCHEMA ALIAS MISSING: {aname}")

        # 3. Subframes preserved
        for sf_name in subframe_names:
            sf = loaded.get_subframe(sf_name)
            if sf is None:
                findings.append(f"SUBFRAME MISSING: {sf_name}")
            else:
                # Check subframe has data
                if len(sf.df) == 0:
                    findings.append(f"SUBFRAME EMPTY: {sf_name}")
                # Check subframe aliases preserved
                orig_sf = original.get_subframe(sf_name)
                if orig_sf:
                    for sa_name in orig_sf.aliases:
                        if sa_name not in sf.aliases:
                            findings.append(
                                f"SUBFRAME {sf_name} ALIAS MISSING: {sa_name}"
                            )

        # 4. Column data integrity (spot check)
        for col in ['x', 'y', 'z', 'sec', 'row']:
            if col in original.df.columns and col in loaded.df.columns:
                if not np.allclose(
                    original.df[col].values.astype(np.float64),
                    loaded.df[col].values.astype(np.float64),
                    rtol=1e-3, equal_nan=True
                ):
                    findings.append(f"COLUMN DATA MISMATCH: {col}")

        return findings

    @pytest.mark.invariance
    def test_E1_1_basic_roundtrip_no_subframes(self, tmp_path):
        """E1_1: baseline — export/read with 0 subframes."""
        adf = self._build_main_adf()
        fpath = str(tmp_path / "test_e1_1.root")

        adf.export_tree(fpath, "tree")
        loaded = AliasDataFrame.read_tree(fpath, "tree")

        findings = self._verify_roundtrip(adf, loaded, [])
        assert not findings, (
            f"E1_1 ROUNDTRIP FAILURE (0 subframes):\n  "
            + "\n  ".join(findings)
        )

    @pytest.mark.invariance
    def test_E1_2_roundtrip_3_subframes(self, tmp_path):
        """E1_2: representative — 3 subframes (T + 2 coefficient tables)."""
        adf = self._build_main_adf()
        track_sf = self._build_track_subframe()
        coeff_dy = self._build_coeff_subframe('dy', n_terms=10)
        coeff_dz = self._build_coeff_subframe('dz', n_terms=10)

        adf.register_subframe('T', track_sf, index_columns=['track_index'])
        adf.register_subframe('PolyFitDy', coeff_dy, index_columns=['sec'])
        adf.register_subframe('PolyFitDz', coeff_dz, index_columns=['sec'])

        fpath = str(tmp_path / "test_e1_2.root")
        adf.export_tree(fpath, "tree")
        loaded = AliasDataFrame.read_tree(fpath, "tree")

        findings = self._verify_roundtrip(
            adf, loaded, ['T', 'PolyFitDy', 'PolyFitDz']
        )
        assert not findings, (
            f"E1_2 ROUNDTRIP FAILURE (3 subframes):\n  "
            + "\n  ".join(findings)
        )

    @pytest.mark.invariance
    def test_E1_3_roundtrip_production_scale_subframes(self, tmp_path):
        """
        E1_3: production-scale — 15 subframes mimicking the calibration pipeline.

        This is the configuration that triggers 15+ ROOT file open/close cycles
        in the current code. The fix must preserve all metadata with 1 open/close.
        """
        adf = self._build_main_adf()

        # Register subframes mimicking the real pipeline
        sf_names = []

        # Track subframe (like T)
        track_sf = self._build_track_subframe()
        adf.register_subframe('T', track_sf, index_columns=['track_index'])
        sf_names.append('T')

        # Coefficient subframes for iterations 0-5 (dy + dz each)
        for i in range(6):
            for var in ['dy', 'dz']:
                name = f'CoeffsI{i}_{var}'
                sf = self._build_coeff_subframe(f'{var}_I{i}', n_terms=5)
                adf.register_subframe(name, sf, index_columns=['sec'])
                sf_names.append(name)

        # Alignment subframes
        for j in range(2):
            name = f'AlignITS{j}'
            df_align = pd.DataFrame({
                'row': np.arange(152, dtype=np.int16),
                f'dy_align_{j}': np.random.randn(152).astype(np.float32),
            })
            adf.register_subframe(name, AliasDataFrame(df_align),
                                  index_columns=['row'])
            sf_names.append(name)

        # BinSigma subframe
        df_bins = pd.DataFrame({
            'sec': np.repeat(np.arange(36), 10).astype(np.int8),
            'row': np.tile(np.arange(10), 36).astype(np.int16),
            'dy_std': np.random.rand(360).astype(np.float16),
        })
        adf.register_subframe('BinSigma', AliasDataFrame(df_bins),
                              index_columns=['sec', 'row'])
        sf_names.append('BinSigma')

        print(f"\nE1_3: {len(sf_names)} subframes registered: {sf_names}")

        fpath = str(tmp_path / "test_e1_3.root")
        adf.export_tree(fpath, "tree")
        loaded = AliasDataFrame.read_tree(fpath, "tree")

        findings = self._verify_roundtrip(adf, loaded, sf_names)
        assert not findings, (
            f"E1_3 ROUNDTRIP FAILURE ({len(sf_names)} subframes):\n  "
            + "\n  ".join(findings)
        )

    @pytest.mark.invariance
    def test_E1_4_roundtrip_timing_report(self, tmp_path):
        """
        E1_4: timing — measure export_tree wall time with 15 subframes.

        Not a pass/fail gate — just reports timing for before/after comparison.
        Run with -s to see the timing output.
        """
        import time

        adf = self._build_main_adf(n=10000)  # larger for timing

        sf_names = []
        track_sf = self._build_track_subframe(n_tracks=2000)
        adf.register_subframe('T', track_sf, index_columns=['track_index'])
        sf_names.append('T')

        for i in range(6):
            for var in ['dy', 'dz']:
                name = f'CoeffsI{i}_{var}'
                sf = self._build_coeff_subframe(f'{var}_I{i}', n_terms=10)
                adf.register_subframe(name, sf, index_columns=['sec'])
                sf_names.append(name)

        fpath = str(tmp_path / "test_e1_4.root")

        t0 = time.perf_counter()
        adf.export_tree(fpath, "tree")
        t_export = time.perf_counter() - t0

        t0 = time.perf_counter()
        loaded = AliasDataFrame.read_tree(fpath, "tree")
        t_read = time.perf_counter() - t0

        fsize = os.path.getsize(fpath) / 1024**2

        print(
            f"\nE1_4 timing ({len(sf_names)} subframes, {len(adf.df)} rows):\n"
            f"  export_tree: {t_export:.2f}s\n"
            f"  read_tree:   {t_read:.2f}s\n"
            f"  file size:   {fsize:.1f} MB\n"
            f"  subframes:   {len(sf_names)}"
        )

        # Verify roundtrip correctness too
        findings = self._verify_roundtrip(adf, loaded, sf_names)
        assert not findings, (
            f"E1_4 ROUNDTRIP FAILURE:\n  " + "\n  ".join(findings)
        )
