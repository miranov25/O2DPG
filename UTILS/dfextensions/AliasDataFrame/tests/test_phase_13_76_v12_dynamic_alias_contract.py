"""PHASE_13_76_ADF B3.2b — tests-first dynamic alias contract checkpoint.

Reference contract:
  PHASE_13_76_ADF_B3_2b_RuntimeResolution_Metadata_ArchitectClarification_v05_FINAL_20260830.md

This file intentionally contains two kinds of tests:
  * ordinary PASS tests for architect decisions already satisfied by current code;
  * strict xfails for independently reproduced DYN defects that are OUTSIDE the
    automatic B3.2b v1.2 implementation scope but must remain executable.

No production behavior is changed by this checkpoint.
"""

import copy

import numpy as np
import pandas as pd
import pytest

try:  # package-style
    from AliasDataFrame.AliasDataFrame import AliasDataFrame, ExpressionNameAbsenceError
except (ImportError, ModuleNotFoundError):  # repo/module-style
    from AliasDataFrame import AliasDataFrame, ExpressionNameAbsenceError


DYN_P0_1 = (
    "DYN-P0-1: redefining a virtual upstream alias does not invalidate an "
    "already-materialized dependent"
)
DYN_P0_2 = (
    "DYN-P0-2: re-registering a subframe does not invalidate materialized "
    "aliases sourced from that subframe"
)
DYN_P1_1 = (
    "DYN-P1-1: rejected indirect-cycle redefinition is not call-atomic"
)
DYN_P2_1 = (
    "DYN-P2-1: malformed scalar syntax can be accepted while inspection "
    "reports the alias healthy"
)


def _base_adf(n=6):
    x = np.arange(1, n + 1, dtype=np.float64)
    return AliasDataFrame(pd.DataFrame({"x": x})), x


def _state_fingerprint(adf):
    """Small deterministic fingerprint of mutable alias-definition state.

    This is intentionally narrower than a future generic M-2 helper.  It is
    sufficient to prove the audited rejected-cycle mutation: schema aliases,
    public alias mapping, and materialized columns/values must all survive a
    rejected mutation unchanged.
    """
    columns = tuple(map(str, adf.df.columns))
    values = {
        c: np.asarray(adf.df[c]).copy()
        for c in columns
    }
    schema_columns = copy.deepcopy(adf._schema.get("columns", {}))
    aliases = copy.deepcopy(dict(adf.aliases))
    return columns, values, schema_columns, aliases


def _assert_same_fingerprint(before, after):
    bcols, bvals, bschema, baliases = before
    acols, avals, aschema, aaliases = after
    assert acols == bcols
    assert aschema == bschema
    assert aaliases == baliases
    assert set(avals) == set(bvals)
    for name in bvals:
        np.testing.assert_array_equal(avals[name], bvals[name])

@pytest.mark.invariance
class TestV12DependencyResolutionContract:
    def test_ev_1_three_level_chain_matches_independent_numpy_oracle(self):
        adf, x = _base_adf()
        adf.add_alias("a", "x*2")
        adf.add_alias("b", "a+1")
        adf.add_alias("c", "b*3")

        got = np.asarray(adf.eval("c"))
        expected = (x * 2.0 + 1.0) * 3.0
        np.testing.assert_array_equal(got, expected)

    def test_ev_4_unresolved_definition_is_legal_and_inspectable(self):
        adf, _ = _base_adf()
        adf.add_alias("q", "nosuch*2")
        adf.add_alias("q2", "S.nosuch*2")

        broken = set(adf.validate_aliases())
        assert "q" in broken
        assert "q2" in broken

    def test_ev_5_unresolved_failure_occurs_at_use_unrelated_eval_stays_healthy(self):
        adf, x = _base_adf()
        adf.add_alias("q", "nosuch*2")

        with pytest.raises(ExpressionNameAbsenceError):
            adf.eval("q")

        np.testing.assert_array_equal(np.asarray(adf.eval("x+1")), x + 1.0)

    def test_ev_6_definition_time_structural_boundary(self):
        adf, _ = _base_adf()

        # Missing dependency is deferred, not rejected at definition time.
        adf.add_alias("missing_ok", "future_name+1")

        with pytest.raises(ValueError):
            adf.add_alias("self_ref", "self_ref+1")

        cyc = AliasDataFrame(pd.DataFrame({"x": [1.0, 2.0]}))
        cyc.add_alias("a", "b+1")
        with pytest.raises(ValueError, match="[Cc]ycle"):
            cyc.add_alias("b", "a+1")

        vec = AliasDataFrame(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
        with pytest.raises(ValueError):
            vec.add_alias(["u", "v", "w"], "a+b, a-b")

@pytest.mark.invariance
class TestV12InvalidationContract:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P0_1)
    def test_ev_2_virtual_upstream_redefine_invalidates_materialized_dependent(self):
        """P2-A exact audited topology: upstream stays virtual, sink is materialized."""
        adf, x = _base_adf()
        adf.add_alias("a", "x*2")
        adf.add_alias("b", "a+1")

        # Default with_dependencies=True + cleanTemporary=True materializes b
        # while removing temporary a.  This is the public reproducer that the
        # older fully-materialized V2/V3 tests miss.
        adf.materialize_aliases(names=["b"])
        assert "a" not in adf.df.columns
        assert "b" in adf.df.columns
        np.testing.assert_array_equal(np.asarray(adf.df["b"]), x * 2.0 + 1.0)

        adf.add_alias("a", "x*100")

        # Architect Decision D: the stale sink may not remain authoritative.
        assert "b" not in adf.df.columns
        np.testing.assert_array_equal(np.asarray(adf.eval("b")), x * 100.0 + 1.0)

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P0_2)
    def test_ev_3_subframe_reregistration_invalidates_sourced_alias(self):
        n = 5
        main = AliasDataFrame(pd.DataFrame({
            "sec": np.arange(n, dtype=np.int64),
            "x": np.arange(n, dtype=np.float64),
        }))
        coeff0 = AliasDataFrame(pd.DataFrame({
            "sec": np.arange(n, dtype=np.int64),
            "c0": np.zeros(n, dtype=np.float64),
        }))
        coeff1_values = np.arange(1, n + 1, dtype=np.float64)
        coeff1 = AliasDataFrame(pd.DataFrame({
            "sec": np.arange(n, dtype=np.int64),
            "c0": coeff1_values,
        }))

        main.register_subframe("Coeff", coeff0, index_columns=["sec"])
        main.add_alias("correction", "Coeff.c0")
        main.materialize_aliases(names=["correction"])
        np.testing.assert_array_equal(np.asarray(main.df["correction"]), 0.0)

        # Deliberately DO NOT redefine correction.  That redundant action is
        # what masks this defect in the older production-pattern test.
        main.register_subframe("Coeff", coeff1, index_columns=["sec"])

        assert "correction" not in main.df.columns
        got = np.asarray(main.eval("correction"))
        np.testing.assert_array_equal(got, coeff1_values)

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P1_1)
    def test_ev_7_rejected_cycle_redefinition_is_state_atomic(self):
        adf, x = _base_adf()
        adf.add_alias("a", "x*2")
        adf.add_alias("b", "a+1")
        adf.materialize_aliases(names=["a", "b"], cleanTemporary=False)

        before = _state_fingerprint(adf)
        with pytest.raises(ValueError, match="[Cc]ycle"):
            adf.add_alias("a", "b+1")
        after = _state_fingerprint(adf)

        _assert_same_fingerprint(before, after)
        np.testing.assert_array_equal(np.asarray(adf.eval("b")), x * 2.0 + 1.0)
        # A rejected mutation must not poison later unrelated mutations.
        adf.add_alias("c", "x+10")
        np.testing.assert_array_equal(np.asarray(adf.eval("c")), x + 10.0)

@pytest.mark.invariance
class TestV12InspectionContract:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P2_1)
    def test_ev_8_malformed_scalar_syntax_is_visible_to_inspection(self):
        """Do not ratify current malformed-syntax acceptance as 'healthy'."""
        adf, _ = _base_adf()
        adf.add_alias("bad_syntax", "x*")
        assert "bad_syntax" in set(adf.validate_aliases())
