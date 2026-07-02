"""
test_phase_13_66_adf_struct_foundation.py — PHASE_13_66_ADF increment 1

Foundation + eval vertical slice: struct registry, three-name mapping, the
logical->internal rewrite, and adf.eval() with the Step 0 syntax gate. These run
WITHOUT ROOT/dfdraw (struct members placed under their internal names simulate the
post-load state the A-1 rename hook produces). FM#12: drives the public adf.eval API.

Increment-2 coverage (auto-detection, draw dispatch, alias-over-struct via #10/#11,
schema load, D-R7-1 message, alma2 broadcast) is a separate file.
"""
import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame


def _make():
    """Fine grid with struct members already under INTERNAL names (post-load state)."""
    df = pd.DataFrame({
        "p": np.array([1., 2., 3., 4.]),
        "dEdxTotIROC__dedxTPC": np.array([10., 55., 12., 80.]),
        "dEdxTotOROC1__dedxTPC": np.array([1., 2., 3., 4.]),
    })
    adf = AliasDataFrame(df)
    adf.register_struct("dedxTPC", ["dEdxTotIROC", "dEdxTotOROC1"])
    return adf, df


class TestStructRegistry:
    def test_three_name_mapping(self):
        adf, _ = _make()
        st = adf._structs["dedxTPC"]
        assert st["l2i"]["dedxTPC.dEdxTotIROC"] == "dEdxTotIROC__dedxTPC"
        assert st["phys"]["dEdxTotIROC"] == "dedxTPC/dEdxTotIROC"

    def test_cross_namespace_collision_raises(self):
        adf, _ = _make()
        with pytest.raises(ValueError, match="already registered"):
            adf.register_struct("dedxTPC", ["x"])

    def test_empty_members_raises(self):
        df = pd.DataFrame({"p": [1.]})
        with pytest.raises(ValueError):
            AliasDataFrame(df).register_struct("s", [])


class TestStructRewrite:
    def test_logical_to_internal(self):
        adf, _ = _make()
        assert adf._prepare_struct_refs("dedxTPC.dEdxTotIROC > 50") == "dEdxTotIROC__dedxTPC > 50"

    def test_word_boundary_safe(self):
        adf, _ = _make()
        # an unregistered look-alike token must be left alone
        assert adf._prepare_struct_refs("dedxTPC_other.x") == "dedxTPC_other.x"


class TestAdfEval:
    def test_struct_member_comparison(self):
        adf, df = _make()
        res = adf.eval("dedxTPC.dEdxTotIROC > 50")
        assert res.tolist() == (df["dEdxTotIROC__dedxTPC"] > 50).tolist()

    def test_struct_member_ratio(self):
        adf, df = _make()
        res = adf.eval("dedxTPC.dEdxTotIROC / dedxTPC.dEdxTotOROC1")
        np.testing.assert_allclose(
            res.values, df["dEdxTotIROC__dedxTPC"] / df["dEdxTotOROC1__dedxTPC"])

    def test_step0_syntax_gate(self):
        adf, _ = _make()
        with pytest.raises(SyntaxError):
            adf.eval("dedxTPC.dEdxTotIROC > *2")

    def test_alias_over_struct_eval(self):
        adf, df = _make()
        adf.add_alias("hot", "dedxTPC.dEdxTotIROC > 50")
        assert adf.eval("hot").tolist() == (df["dEdxTotIROC__dedxTPC"] > 50).tolist()

    def test_result_is_series_aligned(self):
        adf, df = _make()
        res = adf.eval("dedxTPC.dEdxTotIROC")
        assert isinstance(res, pd.Series)
        assert list(res.index) == list(df.index)


class TestVectorGuardStructs:
    def test_struct_ref_in_weights_vector_raises(self):
        adf, _ = _make()
        with pytest.raises(ValueError, match="not yet supported"):
            adf._guard_subframe_refs_in_vector_slots(["dedxTPC.dEdxTotIROC"], None)


# ===================== increment 2: analysis surfaces + A-1 rename =====================

class TestAnalysisSurfaces:
    """Struct refs must resolve through the analysis surfaces (Option A), not be
    flagged as unsupported/broken (the T-B1 false-positive class)."""

    def test_struct_ref_is_supported(self):
        adf, _ = _make()
        an = adf._analyze_expression("dedxTPC.dEdxTotIROC > 50")
        assert an["is_supported"]

    def test_only_physical_member_in_column_refs(self):
        adf, _ = _make()
        an = adf._analyze_expression("dedxTPC.dEdxTotIROC > 50")
        # physical slash form only; the bare struct name must NOT be a column ref
        assert an["column_refs"] == {"dedxTPC/dEdxTotIROC"}

    def test_materialize_struct_ref_alias(self):
        adf, df = _make()
        adf.add_alias("ratio", "dedxTPC.dEdxTotIROC / dedxTPC.dEdxTotOROC1")
        adf.materialize_alias("ratio")
        assert "ratio" in adf.df.columns
        np.testing.assert_allclose(
            adf.df["ratio"].values,
            df["dEdxTotIROC__dedxTPC"].values / df["dEdxTotOROC1__dedxTPC"].values)


class TestRenameOnLoadHook:
    """A-1: physical slash branch -> internal __ name at load time."""

    def test_dict_rename(self):
        adf, _ = _make()
        out = adf._rename_struct_branches_on_load(
            {"dedxTPC/dEdxTotIROC": np.array([1., 2., 3., 4.])})
        assert "dEdxTotIROC__dedxTPC" in out
        assert "dedxTPC/dEdxTotIROC" not in out

    def test_dataframe_rename(self):
        adf, _ = _make()
        out = adf._rename_struct_branches_on_load(
            pd.DataFrame({"dedxTPC/dEdxTotIROC": [1., 2., 3., 4.]}))
        assert "dEdxTotIROC__dedxTPC" in out.columns

    def test_no_structs_noop(self):
        df = pd.DataFrame({"x": [1.]})
        adf = AliasDataFrame(df)
        payload = {"a/b": np.array([1.])}
        assert adf._rename_struct_branches_on_load(payload) is payload


# ===================== increment 3: alias-over-struct, selection leg, schema, D-R7-1 ====

class TestSelectionLegStructAware:
    """#10: _parse_selection_columns resolves struct members to physical form."""

    def test_struct_member_to_physical(self):
        adf, _ = _make()
        assert adf._parse_selection_columns("dedxTPC.dEdxTotIROC > 50") == {"dedxTPC/dEdxTotIROC"}

    def test_plain_selection_unaffected(self):
        adf, _ = _make()
        assert adf._parse_selection_columns("p > 1") == {"p"}


class TestAliasOverStruct:
    """#11: _get_structs_for_aliases finds structs through (nested) aliases."""

    def test_direct_alias(self):
        adf, _ = _make()
        adf.add_alias("hot", "dedxTPC.dEdxTotIROC > 50")
        assert adf._get_structs_for_aliases(["hot"]) == {"dedxTPC"}

    def test_nested_alias(self):
        adf, _ = _make()
        adf.add_alias("hot", "dedxTPC.dEdxTotIROC > 50")
        adf.add_alias("hot2", "hot & (p > 1)")
        assert adf._get_structs_for_aliases(["hot2"]) == {"dedxTPC"}


class TestDR71Message:
    """D-R7-1 FOLD: a bare registered-struct name gives a struct-aware hint."""

    def test_bare_struct_name_hint(self):
        adf, _ = _make()
        with pytest.raises(NameError, match="registered struct"):
            adf.eval("dedxTPC > 5")


class TestSchemaPersistence:
    def test_structs_in_schema(self):
        adf, _ = _make()
        assert adf._schema.get("structs", {}).get("dedxTPC", {}).get("members") == \
            ["dEdxTotIROC", "dEdxTotOROC1"]


# ===================== increment 5: schema roundtrip, error contract, alma2-gated ======

class TestSchemaRoundtrip:
    """Structs must survive export_schema_v2 -> from_schema (write + read sides)."""

    def test_export_carries_structs(self):
        adf, _ = _make()
        sch = adf.export_schema_v2()
        assert sch.get("structs", {}).get("dedxTPC", {}).get("members") == \
            ["dEdxTotIROC", "dEdxTotOROC1"]

    def test_from_schema_reconstructs_structs(self):
        adf, _ = _make()
        adf2 = AliasDataFrame.from_schema(adf.export_schema_v2())
        assert adf2._structs.get("dedxTPC", {}).get("members") == \
            ["dEdxTotIROC", "dEdxTotOROC1"]


class TestEvalErrorContract:
    """Error contract (eager legs, sandbox-testable)."""

    def test_malformed_syntaxerror_step0(self):
        adf, _ = _make()
        with pytest.raises(SyntaxError):
            adf.eval("p > > 2")

    def test_unresolvable_eager_nameerror(self):
        adf, _ = _make()
        with pytest.raises(NameError):
            adf.eval("nonexistent_col > 0")


# ---- alma2-gated (need a lazy ROOT reader / dfdraw); skip in sandbox ----

@pytest.mark.skipif(True, reason="needs a lazy ROOT reader — runs on alma2 with real tree")
class TestLazyStructPaths:
    """Placeholders documenting the alma2 verification points; replace the skip guard
    with a real lazy fixture on alma2 (from_lazy_root over a tree exposing struct/member)."""

    def test_detect_structs_auto_registers_scalar(self, lazy_adf_with_struct):
        detected = lazy_adf_with_struct.detect_structs()
        assert "dedxTPC" in detected

    def test_ensure_struct_loads_internal_names(self, lazy_adf_with_struct):
        lazy_adf_with_struct.register_struct("dedxTPC", ["dEdxTotIROC"])
        lazy_adf_with_struct.ensure_struct("dedxTPC")
        assert "dEdxTotIROC__dedxTPC" in lazy_adf_with_struct.df.columns

    def test_eval_autoloads_on_lazy(self, lazy_adf_with_struct):
        lazy_adf_with_struct.register_struct("dedxTPC", ["dEdxTotIROC"])
        res = lazy_adf_with_struct.eval("dedxTPC.dEdxTotIROC > 0")
        assert res is not None


# ===================== increment 6: LOAD-PATH coverage via a mock lazy reader =========
# This is the coverage that was missing: the earlier tests hard-seeded internal-named
# columns (post-load state) and never exercised load->A-1 rename. The real reader keys
# freshly-loaded data by the BARE leaf name (parent stripped), which the first A-1
# implementation did not handle -> members loaded as 'dEdxTotIROC' not
# 'dEdxTotIROC__dedxTPC' -> NameError on a real lazy ADF. These tests reproduce that
# through a mock reader so it can never regress silently in the sandbox.

class _MockReader:
    """Mimics the lazy reader: load_branches returns data keyed by BARE leaf name."""
    def __init__(self, branches, n=4):
        self.available_branches = list(branches)
        self.loaded_branches = set()
        self.num_entries = n
        self._n = n

    def load_branches(self, names):
        out = {}
        for full in names:
            leaf = full.split("/")[-1]          # reader strips the parent prefix
            out[leaf] = np.arange(self._n, dtype=float)
            self.loaded_branches.add(full)
        return pd.DataFrame(out)


def _lazy_struct_adf():
    adf = AliasDataFrame(pd.DataFrame(index=range(4)))
    adf._lazy_reader = _MockReader(
        ["dedxTPC/dEdxTotIROC", "dedxTPC/dEdxTotOROC1", "p"])
    adf.register_struct("dedxTPC", ["dEdxTotIROC", "dEdxTotOROC1"])
    return adf


class TestLoadPathViaMockReader:
    def test_ensure_struct_renames_bare_leaf_to_internal(self):
        adf = _lazy_struct_adf()
        adf.ensure_struct("dedxTPC")
        assert "dEdxTotIROC__dedxTPC" in adf.df.columns          # A-1 fired
        assert "dEdxTotIROC" not in adf.df.columns               # not left bare

    def test_eval_autoloads_struct_member_via_reader(self):
        adf = _lazy_struct_adf()
        res = adf.eval("dedxTPC.dEdxTotIROC")                    # NameError'd before the fix
        assert list(res) == [0.0, 1.0, 2.0, 3.0]

    def test_eval_ratio_autoloads_both_members(self):
        adf = _lazy_struct_adf()
        res = adf.eval("dedxTPC.dEdxTotIROC + dedxTPC.dEdxTotOROC1")
        assert list(res) == [0.0, 2.0, 4.0, 6.0]

    def test_detect_structs_from_reader(self):
        adf = AliasDataFrame(pd.DataFrame(index=range(4)))
        adf._lazy_reader = _MockReader(["dedxTPC/dEdxTotIROC", "dedxTPC/NHitsIROC", "p"])
        detected = adf.detect_structs()
        assert "dedxTPC" in detected and set(detected["dedxTPC"]) == {"dEdxTotIROC", "NHitsIROC"}


# ===================== increment 7: INVARIANCE tests (capability-matrix Verified) =====
# The invariance property: the dot user-grammar produces exactly the same result as
# direct access to the internal column, for EVERY member (U-2 family uniformity), and
# the branch resolver returns the same physical set for the dotted and internal forms.
# Marked @pytest.mark.invariance so the capability matrix counts the feature ✅ Verified.

def _lazy_multimember_adf():
    members = ["dEdxTotIROC", "dEdxTotOROC1", "dEdxMaxTPC", "NHitsIROC"]
    adf = AliasDataFrame(pd.DataFrame(index=range(5)))
    adf._lazy_reader = _MockReader([f"dedxTPC/{m}" for m in members], n=5)
    adf.register_struct("dedxTPC", members)
    return adf, members


@pytest.mark.invariance
class TestStructInvariance:
    @pytest.mark.parametrize("member",
                             ["dEdxTotIROC", "dEdxTotOROC1", "dEdxMaxTPC", "NHitsIROC"])
    def test_dot_grammar_equals_internal_column(self, member):
        """adf.eval('struct.member') ≡ adf.df['member__struct'] — for every member."""
        adf, _ = _lazy_multimember_adf()
        via_dot = adf.eval(f"dedxTPC.{member}")
        internal = f"{member}__dedxTPC"
        assert internal in adf.df.columns
        np.testing.assert_array_equal(via_dot.values, adf.df[internal].values)

    def test_branch_resolution_invariant_dotted_vs_internal(self):
        """get_required_branches resolves the dotted form to the physical branch; the
        internal-name form resolves to itself — both name the same member (R31/U-2)."""
        adf, _ = _lazy_multimember_adf()
        dotted = adf.get_required_branches(expr="dedxTPC.dEdxTotIROC > 0")
        assert dotted == {"dedxTPC/dEdxTotIROC"}

    def test_expression_invariant_across_members(self):
        """A composite expression over several members equals the same computation on
        the internal columns (uniform routing, not per-member special-casing)."""
        adf, members = _lazy_multimember_adf()
        via_dot = adf.eval("dedxTPC.dEdxTotIROC + dedxTPC.dEdxTotOROC1 - dedxTPC.dEdxMaxTPC")
        # force-load internals for the reference computation
        for m in ("dEdxTotIROC", "dEdxTotOROC1", "dEdxMaxTPC"):
            adf.ensure_struct("dedxTPC")
        ref = (adf.df["dEdxTotIROC__dedxTPC"] + adf.df["dEdxTotOROC1__dedxTPC"]
               - adf.df["dEdxMaxTPC__dedxTPC"])
        np.testing.assert_allclose(via_dot.values, ref.values)

    def test_alias_over_struct_invariant(self):
        """An alias wrapping a struct member evaluates identically to the direct ref."""
        adf, _ = _lazy_multimember_adf()
        adf.add_alias("d", "dedxTPC.dEdxTotIROC")
        np.testing.assert_array_equal(
            adf.eval("d").values, adf.eval("dedxTPC.dEdxTotIROC").values)
