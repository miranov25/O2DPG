"""
Phase 13.20.ADF — E2 Fix A Verification Tests

PURPOSE: Verify the export_tree metadata batching fix (Phase 13.20.ADF Fix A)
is correct and doesn't break anything.

These tests complement E1 (roundtrip correctness) with fix-specific checks:
- E2_1: TFile.Open is called exactly 1 time per export_tree (not N+1)
- E2_2: Nested subframes (2 levels deep) survive roundtrip (latent bug fix)
- E2_3: Files written by new code readable by current read_tree (backward compat)
- E2_4: Standalone _write_metadata_to_root still works (backward compat for external callers)

Requires ROOT (import ROOT). Skip if not available.
"""

import os
import sys
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest import mock

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


def _build_adf(n=200, seed=1320):
    """Small ADF for testing."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'x': rng.uniform(85.0, 245.0, n).astype(np.float32),
        'y': rng.uniform(-50.0, 50.0, n).astype(np.float32),
        'z': rng.uniform(-250.0, 250.0, n).astype(np.float32),
        'dy': rng.normal(0.0, 0.5, n).astype(np.float16),
        'sec': rng.integers(0, 36, n).astype(np.int8),
        'row': rng.integers(0, 152, n).astype(np.int16),
    })
    adf = AliasDataFrame(df)
    adf.add_alias('r', 'sqrt(x**2 + y**2)', dtype=np.float32)
    adf.add_alias('drift', '1 - abs(z)/250', dtype=np.float16)
    return adf


def _build_coeff_sf(prefix, n_sectors=36, n_terms=5, seed=42):
    """Small coefficient subframe."""
    rng = np.random.default_rng(seed)
    cols = {'sec': np.arange(n_sectors, dtype=np.int8)}
    for i in range(n_terms):
        cols[f'{prefix}_c{i}'] = rng.normal(0, 0.01, n_sectors).astype(np.float32)
    adf = AliasDataFrame(pd.DataFrame(cols))
    adf.add_alias(f'{prefix}_sum', f'{prefix}_c0 + {prefix}_c1')
    return adf


@pytest.mark.skipif(not HAS_ROOT, reason="ROOT not available")
@pytest.mark.skipif(not HAS_ADF, reason="AliasDataFrame not available")
class TestE2ExportTreeFixA:
    """E2_1..E2_4 — Fix A specific verification tests."""

    @pytest.mark.invariance
    def test_E2_1_single_tfile_open_per_export(self, tmp_path):
        """
        E2_1: TFile.Open must be called exactly 1 time per export_tree call.

        Pre-fix: N+1 calls (1 for main + 1 per subframe).
        Post-fix: exactly 1 call via _write_all_metadata_to_root.

        Uses mock.patch to count ROOT.TFile.Open invocations.
        """
        adf = _build_adf()
        # Register 5 subframes
        for i in range(5):
            sf = _build_coeff_sf(f'coeff{i}', seed=i)
            adf.register_subframe(f'SF{i}', sf, index_columns=['sec'])

        fpath = str(tmp_path / "test_e2_1.root")

        # Count TFile.Open calls
        original_open = ROOT.TFile.Open
        open_count = [0]

        def counting_open(*args, **kwargs):
            open_count[0] += 1
            return original_open(*args, **kwargs)

        with mock.patch.object(ROOT.TFile, 'Open', side_effect=counting_open):
            adf.export_tree(fpath, "tree")

        print(f"\nE2_1: ROOT.TFile.Open called {open_count[0]} time(s) "
              f"for 5 subframes (expected: 1)")

        assert open_count[0] == 1, (
            f"E2_1: export_tree with 5 subframes called ROOT.TFile.Open "
            f"{open_count[0]} times, expected exactly 1. "
            f"Fix A should batch all metadata writes into a single TFile.Open."
        )

        # Verify the file is actually readable
        loaded = AliasDataFrame.read_tree(fpath, "tree")
        assert len(loaded.df) == len(adf.df)

    @pytest.mark.invariance
    def test_E2_2_nested_subframe_roundtrip(self, tmp_path):
        """
        E2_2: Nested subframes (2 levels deep) must survive roundtrip.

        Pre-fix: latent crash — recursive export_tree passed uproot file
        object to ROOT.TFile.Open when a subframe had nested subframes.
        Post-fix: _write_all_data_to_uproot handles recursion without
        mixing uproot and ROOT file handles.
        """
        # Build main ADF
        adf = _build_adf()

        # Build a subframe that itself has a nested subframe
        sf_outer = _build_coeff_sf('outer', n_sectors=36, seed=100)

        # Nested: a small lookup table inside the coefficient subframe
        df_inner = pd.DataFrame({
            'sec': np.arange(36, dtype=np.int8),
            'correction': np.random.randn(36).astype(np.float32),
        })
        sf_inner = AliasDataFrame(df_inner)
        sf_inner.add_alias('corr_abs', 'abs(correction)', dtype=np.float32)

        # Register nested subframe on outer
        sf_outer.register_subframe('Inner', sf_inner, index_columns=['sec'])

        # Register outer on main
        adf.register_subframe('Outer', sf_outer, index_columns=['sec'])

        # Also register a flat subframe for comparison
        sf_flat = _build_coeff_sf('flat', seed=200)
        adf.register_subframe('Flat', sf_flat, index_columns=['sec'])

        fpath = str(tmp_path / "test_e2_2_nested.root")
        adf.export_tree(fpath, "tree")
        loaded = AliasDataFrame.read_tree(fpath, "tree")

        findings = []

        # Check main aliases
        for alias in ['r', 'drift']:
            if alias not in loaded.aliases:
                findings.append(f"MAIN ALIAS MISSING: {alias}")

        # Check outer subframe
        sf_loaded_outer = loaded.get_subframe('Outer')
        if sf_loaded_outer is None:
            findings.append("OUTER SUBFRAME MISSING")
        else:
            if 'outer_sum' not in sf_loaded_outer.aliases:
                findings.append("OUTER ALIAS 'outer_sum' MISSING")

            # Check nested subframe (2 levels deep)
            sf_loaded_inner = sf_loaded_outer.get_subframe('Inner')
            if sf_loaded_inner is None:
                findings.append("NESTED SUBFRAME 'Inner' MISSING (2 levels deep)")
            else:
                if len(sf_loaded_inner.df) != 36:
                    findings.append(
                        f"NESTED SUBFRAME DATA: expected 36 rows, "
                        f"got {len(sf_loaded_inner.df)}"
                    )
                if 'corr_abs' not in sf_loaded_inner.aliases:
                    findings.append("NESTED ALIAS 'corr_abs' MISSING")

        # Check flat subframe
        sf_loaded_flat = loaded.get_subframe('Flat')
        if sf_loaded_flat is None:
            findings.append("FLAT SUBFRAME MISSING")

        print(f"\nE2_2 nested subframe roundtrip: "
              f"{'PASS' if not findings else 'FAIL'}")
        if findings:
            for f_ in findings:
                print(f"  {f_}")

        assert not findings, (
            f"E2_2 NESTED ROUNDTRIP FAILURE:\n  "
            + "\n  ".join(findings)
        )

    @pytest.mark.invariance
    def test_E2_3_read_tree_backward_compatibility(self, tmp_path):
        """
        E2_3: Files written by new code must be readable by current read_tree.

        This is a backward compatibility test. The new export code writes
        the same metadata format — just from a single TFile.Open instead
        of N+1. read_tree should not notice the difference.
        """
        adf = _build_adf()
        sf1 = _build_coeff_sf('dy', seed=10)
        sf2 = _build_coeff_sf('dz', seed=20)
        adf.register_subframe('PolyDy', sf1, index_columns=['sec'])
        adf.register_subframe('PolyDz', sf2, index_columns=['sec'])

        fpath = str(tmp_path / "test_e2_3.root")
        adf.export_tree(fpath, "tree")

        # Read back and verify everything
        loaded = AliasDataFrame.read_tree(fpath, "tree")

        # Main data
        assert len(loaded.df) == len(adf.df), "Row count mismatch"

        # Aliases
        assert 'r' in loaded.aliases, "Alias 'r' missing"
        assert 'drift' in loaded.aliases, "Alias 'drift' missing"

        # Subframes
        for sf_name in ['PolyDy', 'PolyDz']:
            sf = loaded.get_subframe(sf_name)
            assert sf is not None, f"Subframe '{sf_name}' missing"
            assert len(sf.df) == 36, f"Subframe '{sf_name}' row count wrong"

        # Schema roundtrip
        assert hasattr(loaded, '_schema'), "Schema missing"

        # Materialize a loaded alias to verify it evaluates correctly
        loaded.materialize_alias('r')
        adf.materialize_alias('r')
        assert np.allclose(
            loaded.df['r'].values.astype(np.float64),
            adf.df['r'].values.astype(np.float64),
            rtol=1e-3
        ), "Materialized alias 'r' value mismatch after roundtrip"

        print("\nE2_3 backward compatibility: PASS")

    @pytest.mark.invariance
    def test_E2_4_standalone_write_metadata_to_root(self, tmp_path):
        """
        E2_4: _write_metadata_to_root (standalone, backward compat) still works.

        External code may call _write_metadata_to_root directly.
        After refactoring, it should delegate to _write_metadata_to_tree
        and produce identical results.
        """
        adf = _build_adf()
        adf.add_alias('test_alias', 'x + y', dtype=np.float32)

        fpath = str(tmp_path / "test_e2_4.root")

        # First write the data via uproot (without metadata)
        import uproot
        with uproot.recreate(fpath) as f:
            export_cols = [c for c in adf.df.columns]
            dtype_casts = {c: np.float32 for c in export_cols
                          if adf.df[c].dtype == np.float16}
            export_df = adf.df[export_cols].astype(dtype_casts)
            f["tree"] = {c: export_df[c].values for c in export_df.columns}

        # Now call _write_metadata_to_root directly (the backward-compat path)
        adf._write_metadata_to_root(fpath, "tree")

        # Read back and verify metadata survived
        f = ROOT.TFile.Open(fpath)
        tree = f.Get("tree")
        assert tree is not None, "Tree not found"

        # Check TTree alias was set
        alias_val = tree.GetAlias("test_alias")
        assert alias_val, (
            "TTree alias 'test_alias' not found — "
            "_write_metadata_to_root backward compat broken"
        )

        # Check UserInfo metadata
        user_info = tree.GetUserInfo()
        assert user_info.GetSize() > 0, "UserInfo is empty"

        import json
        meta_str = user_info.At(0).GetName()
        meta = json.loads(meta_str)
        assert 'aliases' in meta, "aliases key missing from metadata"
        assert 'test_alias' in meta['aliases'], (
            "'test_alias' not in metadata aliases"
        )

        f.Close()
        print("\nE2_4 standalone _write_metadata_to_root: PASS")
