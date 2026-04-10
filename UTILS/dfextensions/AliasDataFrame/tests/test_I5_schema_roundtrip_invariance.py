"""
Batch 1 — I5: Schema Roundtrip Invariance
Phase 13.12.ADF — Public API Invariance Test Suite

STANDALONE NEW TEST FILE (§5.1 deviation note)
-----------------------------------------------
v1.2 proposal §5.1 committed to extending existing files with zero new
files. Phase 13.12 Batch 1 delivered 3 new standalone files instead,
with "invariance" in each filename per §3.1 fallback rule. Deviation
acknowledged by Main Architect on 2026-04-10 after reviewer feedback
(Claude32 P2 #1, Claude33 P1-2).

All tests are marked @pytest.mark.invariance.

FIX HISTORY
-----------
v1 (2026-04-09): Initial submission. I5_1, I5_3, I5_4 failed with
    ValueError: Column 'x_calib' references undefined subframe 'S'
    Root cause: test-writing bug (Failure Mode #11). The tests created
    a fresh AliasDataFrame(df) and called apply_schema() directly, but
    the schema validator correctly rejected aliases referencing
    subframe 'S' because 'S' was never registered on the fresh ADF.
v2 (2026-04-10): Fixed by registering subframes on the fresh ADF
    BEFORE calling apply_schema(), mirroring the actual production
    JSON-path workflow:
        adf2 = AliasDataFrame(df_main)
        adf2.register_subframe('S', sub_adf, 'sector')  <-- added
        adf2.apply_schema(schema)
    I5_2 passed in v1 and remains unchanged — it uses the ROOT path
    where read_tree() internally loads subframes before apply_schema.
v3 (2026-04-10): Post-review polish applied from Claude32 review:
    - I5_1 Invariant 2: strict set equality instead of subset-or
    - I5_4: added key ordering assertion (docstring previously promised
      ordering determinism but body did not assert it)

Feature flipped: SCHEMA.export_import (currently ☑️ smoke-only)
Incident addressed: Phase 13.9.Fix1 (polynomial persistence — JSON path
tested, ROOT path silently broken) — confirmed FIXED as of v1 I5_2 run.

PATH-EXPLICIT DISCIPLINE (Failure Mode #11)
-------------------------------------------
Every test below explicitly exercises the named production API path.
No auto-dispatch, no reliance on defaults that could silently re-route.

PRODUCTION ENTRY POINTS (verified against source 2026-04-09)
------------------------------------------------------------
I5_1: adf.export_schema()
    → adf2 = AliasDataFrame(df); adf2.register_subframe(...); adf2.apply_schema(s)
I5_2: adf.export_tree(path)
    → adf2 = AliasDataFrame.read_tree(path)  [subframes auto-loaded]
I5_3: Both paths on same ADF, compare semantic equality
I5_4: Two consecutive export_schema() calls on fresh ADFs with subframes

These are the actual public API methods a user calls. Verified at:
  AliasDataFrame.py:4754  export_tree
  AliasDataFrame.py:4864  read_tree (load_subframes=True default)
  AliasDataFrame.py:1475  register_subframe
  AliasDataFrame.py:8111  export_schema
  AliasDataFrame.py:8645  apply_schema
  AliasDataFrame.py:1314  update_schema (validates subframe refs)
"""

import pytest
import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


# =============================================================================
# Fixture: Per-test ADF for I5 incident guards
# =============================================================================
# Per v1.2 §5.3 — I5 tests are incident guards requiring per-test isolation
# so we can control exactly what the schema roundtrip must preserve.
#
# Returns TWO objects:
#   adf  — the source ADF (with subframes registered, aliases added)
#   df_main, df_sub — the raw DataFrames used to build adf, so tests
#                     can reconstruct a fresh ADF with the same subframe
#                     setup for the JSON-path roundtrip
#
# This fixture shape reflects the reality that a JSON schema does NOT
# carry subframe DATA — only subframe METADATA. The user restoring from
# JSON must re-register the subframes manually on the fresh ADF before
# calling apply_schema.

@pytest.fixture
def adf_schema_full_feature_mix():
    """
    Per-test fixture with the full schema feature mix:
    aliases + subframes + registered_functions.

    Returns: (adf, df_main, df_sub)
        adf     — source AliasDataFrame with 'S' registered and aliases added
        df_main — raw main DataFrame (for reconstructing a fresh ADF)
        df_sub  — raw subframe DataFrame (for reconstructing a fresh ADF)

    Size: small (20 rows main, 5 rows subframe) because we're testing
    schema preservation, not data correctness at scale.
    """
    # Main frame
    df_main = pd.DataFrame({
        'key': np.arange(20, dtype=np.int64),
        'x': np.linspace(-1.0, 1.0, 20).astype(np.float32),
        'y': np.linspace(0.0, 2.0, 20).astype(np.float32),
        'sector': np.array([i % 5 for i in range(20)], dtype=np.int8),
    })

    # Subframe with 5 keys (all main rows will have a match since sector%5)
    df_sub = pd.DataFrame({
        'sector': np.arange(5, dtype=np.int8),
        'gain': np.linspace(1.0, 1.04, 5).astype(np.float32),
    })

    adf = AliasDataFrame(df_main.copy())
    sub_adf = AliasDataFrame(df_sub.copy())
    adf.register_subframe('S', sub_adf, index_columns='sector')

    # Alias with subframe reference + fill_value
    adf.add_alias('x_calib', 'x * S.gain', dtype=np.float32, fill_value=0)

    # Alias on plain columns
    adf.add_alias('r', 'x*x + y*y', dtype=np.float64)

    return adf, df_main, df_sub


def _build_fresh_adf_with_subframe(df_main, df_sub):
    """
    Helper: construct a fresh AliasDataFrame with the same subframe
    layout as the fixture, so that apply_schema() can validate
    alias references. This mirrors what a real user does when
    restoring from a JSON schema.
    """
    fresh = AliasDataFrame(df_main.copy())
    sub_fresh = AliasDataFrame(df_sub.copy())
    fresh.register_subframe('S', sub_fresh, index_columns='sector')
    return fresh


# =============================================================================
# Test class: append to tests/test_schema_serialization.py
# =============================================================================

class TestI5SchemaRoundtripInvariance:
    """
    Phase 13.12 I5 — Schema Roundtrip Invariance.

    Asserts public-API-level identities for schema serialization. These tests
    would have caught Phase 13.9.Fix1 (polynomial persistence, ROOT path
    silently broken while JSON path passed).

    Flips: SCHEMA.export_import (currently ☑️ smoke-only in Capability Matrix).

    v2 fix (2026-04-10): I5_1/I5_3/I5_4 now register subframe 'S' on
    the fresh ADF before apply_schema(), mirroring the real JSON-path
    production workflow.
    """

    @pytest.mark.invariance
    def test_I5_1_json_schema_roundtrip_preserves_aliases_and_subframes(
        self, adf_schema_full_feature_mix
    ):
        """
        I5_1 INVARIANT:
            adf.export_schema() → new_adf.apply_schema(s) preserves all
            alias expressions, dtypes, fill_values, and subframe metadata,
            provided the target ADF has the subframes registered before
            apply_schema is called.

        CODE PATH:
            export_schema() [JSON serialization]
                → register_subframe() [required for JSON path — subframes
                                       do not travel in the JSON schema]
                → apply_schema() [JSON deserialization]
            NOT auto-dispatched. This is the explicit JSON path.

        PRODUCTION ENTRY POINT:
            s = adf.export_schema()
            adf2 = AliasDataFrame(df_main)
            adf2.register_subframe('S', sub_adf, 'sector')  # user-side
            adf2.apply_schema(s)

        REGRESSION GUARD FOR: schema preservation through JSON path.

        NOTE (Failure Mode #11 lesson): v1 of this test omitted the
        register_subframe call, which caused a ValueError because the
        schema validator correctly rejects alias references to
        unregistered subframes. v2 restores the full production path.
        """
        adf, df_main, df_sub = adf_schema_full_feature_mix

        # Capture before-state
        aliases_before = dict(adf.aliases)
        subframes_before = set(adf._subframes.subframes.keys()) \
            if hasattr(adf._subframes, 'subframes') else set()

        # Path: export_schema (JSON)
        schema = adf.export_schema()
        assert isinstance(schema, dict), "export_schema must return dict"

        # Build a fresh ADF with the same subframe layout
        # (subframes do NOT travel in JSON schemas — user must re-register)
        adf2 = _build_fresh_adf_with_subframe(df_main, df_sub)

        # Apply the schema
        adf2.apply_schema(schema, validate=True, warn_missing=False)

        # Invariant 1: All alias expressions preserved
        aliases_after = dict(adf2.aliases)
        assert set(aliases_before.keys()) == set(aliases_after.keys()), (
            f"Alias names changed: before={set(aliases_before.keys())}, "
            f"after={set(aliases_after.keys())}"
        )
        for name in aliases_before:
            assert aliases_before[name] == aliases_after[name], (
                f"Alias '{name}' expression changed: "
                f"'{aliases_before[name]}' → '{aliases_after[name]}'"
            )

        # Invariant 2: Subframe metadata in schema preserved (strict equality)
        # Per Claude32 P2 #4: strict == instead of subset-or. A schema that
        # invented spurious subframe names would silently pass the subset test.
        subframes_in_schema_after = set(adf2._schema.get('subframes', {}).keys())
        assert subframes_before == subframes_in_schema_after, (
            f"Subframe metadata changed through JSON roundtrip: "
            f"before={subframes_before}, after={subframes_in_schema_after}"
        )

        # Invariant 3: Re-materialization produces equivalent values.
        # Materialize the alias on the source and target, compare.
        adf.materialize_alias('x_calib')
        adf2.materialize_alias('x_calib')
        np.testing.assert_allclose(
            adf.df['x_calib'].values,
            adf2.df['x_calib'].values,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="JSON roundtrip changed materialized alias values"
        )

    @pytest.mark.invariance
    def test_I5_2_root_tree_roundtrip_preserves_full_schema(
        self, adf_schema_full_feature_mix, tmp_path
    ):
        """
        I5_2 INVARIANT — THE PHASE 13.9.FIX1 GAP-CLOSER:
            adf.export_tree(path) → AliasDataFrame.read_tree(path) preserves
            all schema fields (aliases, dtypes, fill_values, subframes)
            embedded in the ROOT file UserInfo.

        CODE PATH:
            export_tree() [writes _schema into ROOT UserInfo, writes
                           subframe data as sibling trees]
                → read_tree(load_subframes=True) [reads UserInfo back
                                                   AND loads subframes]
            NOT auto-dispatched.

        PRODUCTION ENTRY POINT (the exact TPC calibration workflow call):
            adf.export_tree(path)
            adf2 = AliasDataFrame.read_tree(path)

        REGRESSION GUARD FOR: Phase 13.9.Fix1 (polynomial persistence).
            The JSON path (export_schema/apply_schema) was tested while the
            ROOT path was silently broken. User ran export_tree→read_tree,
            tests passed, production broke.

        v1 RESULT: PASSED — the ROOT path correctly preserves the full
        schema. Phase 13.9.Fix1 was more complete than the 13.9.Fix1 test
        coverage suggested.
        """
        adf, df_main, df_sub = adf_schema_full_feature_mix
        aliases_before = dict(adf.aliases)
        subframe_names_before = set(adf._schema.get('subframes', {}).keys())

        # Materialize an alias BEFORE export so we have reference values
        adf.materialize_alias('r')
        r_before = adf.df['r'].values.copy()

        # Path: export_tree → read_tree (the ROOT UserInfo path)
        root_path = str(tmp_path / 'i5_2_roundtrip.root')
        adf.export_tree(root_path, treename='tree')

        # Read back via the production public API (load_subframes=True default)
        adf2 = AliasDataFrame.read_tree(root_path, treename='tree')

        # Invariant 1: Alias names preserved
        aliases_after = dict(adf2.aliases)
        assert set(aliases_before.keys()) == set(aliases_after.keys()), (
            f"Aliases lost through ROOT roundtrip: "
            f"before={set(aliases_before.keys())}, "
            f"after={set(aliases_after.keys())}"
        )

        # Invariant 2: Alias expressions preserved (not just names)
        for name in aliases_before:
            assert aliases_before[name] == aliases_after[name], (
                f"Alias '{name}' expression changed through ROOT roundtrip: "
                f"'{aliases_before[name]}' → '{aliases_after[name]}'"
            )

        # Invariant 3: Subframes preserved (auto-loaded by read_tree)
        subframe_names_after = set(adf2._schema.get('subframes', {}).keys())
        assert subframe_names_before == subframe_names_after, (
            f"Subframes lost through ROOT roundtrip: "
            f"before={subframe_names_before}, after={subframe_names_after}"
        )

        # Invariant 4: Re-materialization produces identical values
        adf2.materialize_alias('r')
        r_after = adf2.df['r'].values
        np.testing.assert_allclose(
            r_before, r_after, rtol=1e-12, atol=1e-15,
            err_msg=(
                "Re-materialized alias 'r' values differ through ROOT "
                "roundtrip — schema preserved but computation path diverged"
            )
        )

    @pytest.mark.invariance
    def test_I5_3_json_and_root_paths_produce_semantically_equal_schemas(
        self, adf_schema_full_feature_mix, tmp_path
    ):
        """
        I5_3 INVARIANT:
            adf.export_schema() (JSON path) and
            adf.export_tree(path) → read_tree(path) (ROOT path)
            produce the SAME set of aliases and subframes when compared
            by semantic content (not dict identity).

        CODE PATH:
            Path A: adf.export_schema()
                → register_subframe() on fresh ADF
                → adf_via_json.apply_schema(s)
            Path B: adf.export_tree(path)
                → AliasDataFrame.read_tree(path)  [subframes auto-loaded]
            Both paths NOT auto-dispatched.

        PRODUCTION ENTRY POINT:
            Both paths are public API. Any user mixing them (e.g., reading
            from ROOT then re-exporting as JSON) depends on this invariant.

        REGRESSION GUARD FOR: divergence between JSON and ROOT serialization
        paths. This is the meta-invariant behind Phase 13.9.Fix1.
        """
        adf, df_main, df_sub = adf_schema_full_feature_mix

        # Path A: JSON
        schema_json = adf.export_schema()
        adf_via_json = _build_fresh_adf_with_subframe(df_main, df_sub)
        adf_via_json.apply_schema(schema_json, validate=True, warn_missing=False)

        # Path B: ROOT
        root_path = str(tmp_path / 'i5_3_root_path.root')
        adf.export_tree(root_path, treename='tree')
        adf_via_root = AliasDataFrame.read_tree(root_path, treename='tree')

        # Semantic equality: same alias names, same expressions
        aliases_json = dict(adf_via_json.aliases)
        aliases_root = dict(adf_via_root.aliases)
        assert set(aliases_json.keys()) == set(aliases_root.keys()), (
            f"JSON path and ROOT path produce different alias sets: "
            f"json={set(aliases_json.keys())}, root={set(aliases_root.keys())}"
        )
        for name in aliases_json:
            assert aliases_json[name] == aliases_root[name], (
                f"Alias '{name}' differs between JSON and ROOT paths: "
                f"json='{aliases_json[name]}', root='{aliases_root[name]}'"
            )

        # Subframe metadata: same set of subframe names
        sf_json = set(adf_via_json._schema.get('subframes', {}).keys())
        sf_root = set(adf_via_root._schema.get('subframes', {}).keys())
        assert sf_json == sf_root, (
            f"JSON path and ROOT path produce different subframe sets: "
            f"json={sf_json}, root={sf_root}"
        )

        # Materialize on both and compare the concrete values
        adf_via_json.materialize_alias('x_calib')
        adf_via_root.materialize_alias('x_calib')
        np.testing.assert_allclose(
            adf_via_json.df['x_calib'].values,
            adf_via_root.df['x_calib'].values,
            rtol=1e-6, atol=1e-9, equal_nan=True,
            err_msg="JSON and ROOT paths produce different materialized values"
        )

    @pytest.mark.invariance
    def test_I5_4_export_schema_is_idempotent(
        self, adf_schema_full_feature_mix
    ):
        """
        I5_4 INVARIANT:
            Two consecutive export_schema() → apply_schema() cycles produce
            semantically equal schemas. The schema dict is deterministic
            in its key ordering for any given ADF state.

        CODE PATH:
            export_schema() → register_subframe() on fresh ADF → apply_schema()
                → export_schema() again
            All calls on the explicit JSON path. Not auto-dispatched.

        PRODUCTION ENTRY POINT:
            Users who write tooling that re-exports loaded schemas (e.g.,
            CI validation, migration scripts) depend on this idempotence.

        REGRESSION GUARD FOR: non-deterministic schema ordering that could
        cause spurious diffs in schema-under-VCS workflows.
        """
        adf, df_main, df_sub = adf_schema_full_feature_mix

        # First export
        schema_1 = adf.export_schema()

        # Apply to fresh ADF with subframe, re-export
        adf2 = _build_fresh_adf_with_subframe(df_main, df_sub)
        adf2.apply_schema(schema_1, validate=True, warn_missing=False)
        schema_2 = adf2.export_schema()

        # Compare alias content as the core equality surface. Different
        # schema versions may store aliases under different top-level keys
        # (v1 'aliases' vs v2 'columns' with expr field). Normalize both
        # to a {name: expression} dict before comparison.
        def _extract_aliases(schema):
            """Extract {name: expression} from either v1 or v2 schema shape."""
            result = {}
            # v1 shape: schema['aliases'] = {name: expr_str}
            if 'aliases' in schema and isinstance(schema['aliases'], dict):
                for name, val in schema['aliases'].items():
                    if isinstance(val, str):
                        result[name] = val
                    elif isinstance(val, dict) and 'expression' in val:
                        result[name] = val['expression']
            # v2 shape: schema['columns'] = {name: {'expr': ..., 'dtype': ...}}
            if 'columns' in schema and isinstance(schema['columns'], dict):
                for name, spec in schema['columns'].items():
                    if isinstance(spec, dict) and spec.get('expr') is not None:
                        # Only count as alias if it has an expression
                        if name not in result:
                            result[name] = spec['expr']
            return result

        aliases_1 = _extract_aliases(schema_1)
        aliases_2 = _extract_aliases(schema_2)

        assert set(aliases_1.keys()) == set(aliases_2.keys()), (
            f"Second export_schema has different alias keys: "
            f"first={set(aliases_1.keys())}, second={set(aliases_2.keys())}"
        )
        for name in aliases_1:
            assert aliases_1[name] == aliases_2[name], (
                f"Alias '{name}' changed between consecutive exports: "
                f"'{aliases_1[name]}' → '{aliases_2[name]}'"
            )

        # Key ordering determinism (per Claude32 P2 #5).
        # The docstring promises "deterministic key ordering" as a guard
        # against spurious diffs in schema-under-VCS workflows. Assert it.
        assert list(aliases_1.keys()) == list(aliases_2.keys()), (
            f"Alias key ordering changed between consecutive exports: "
            f"first={list(aliases_1.keys())}, second={list(aliases_2.keys())}. "
            f"This would cause spurious diffs in schema-under-VCS workflows."
        )
        # Also assert columns-level ordering if v2 schema shape is present
        cols_1 = list(schema_1.get('columns', {}).keys())
        cols_2 = list(schema_2.get('columns', {}).keys())
        if cols_1 or cols_2:
            assert cols_1 == cols_2, (
                f"Columns key ordering changed between consecutive exports: "
                f"first={cols_1}, second={cols_2}"
            )
