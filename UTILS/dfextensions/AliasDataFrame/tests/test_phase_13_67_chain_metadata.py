"""PHASE_13_67_ADF — chain metadata recovery (additive/opt-in). These tests use the
REAL adf_metadata structure (subframes=list of names, subframe_indices=dict,
column_dtypes, aliases) per adf_metadata_compat.read_adf_metadata — NOT an invented
shape. Logic is sandbox-testable; the full chain read over real ROOT files is alma2."""
import pandas as pd
import pytest
from AliasDataFrame import AliasDataFrame
from exceptions import ChainMetadataCompatibilityError


def _real_meta():
    return {"_source": "userinfo",
            "subframes": ["R", "AlignDzITS5"],
            "subframe_indices": {"R": {"index_columns": ["sector"]},
                                 "AlignDzITS5": {"index_columns": ["sector", "row"]}},
            "aliases": {"d": "x/y"},
            "column_dtypes": {"a": "float64"},
            "raw": {}}


class TestNormalizeRealStructure:
    def test_subframes_list_no_crash(self):
        n = AliasDataFrame._normalize_chain_meta(_real_meta())
        assert n["subframes"]["R"]["index_columns"] == ["sector"]
        assert n["aliases"] == {"d": "x/y"}

    def test_names_only_is_valid_sparse_not_refused(self):
        n = AliasDataFrame._normalize_chain_meta(
            {"schema_source": "names_only", "subframes": ["X"],
             "aliases": {}, "column_dtypes": {}})
        assert n["aliases"] == {} and n["dtypes"] == {}
        assert AliasDataFrame._check_chain_metadata_compatibility(
            [{"schema_source": "names_only", "subframes": ["X"], "aliases": {}, "column_dtypes": {}},
             {"schema_source": "names_only", "subframes": ["X"], "aliases": {}, "column_dtypes": {}}],
            ["f0", "f1"]) is not None

    def test_none_metadata(self):
        assert AliasDataFrame._normalize_chain_meta(None) is None


class TestComparatorRealStructure:
    def test_identical_ok(self):
        assert AliasDataFrame._check_chain_metadata_compatibility(
            [_real_meta(), _real_meta()], ["f0", "f1"]) is not None

    def test_all_bare_proceeds(self):
        assert AliasDataFrame._check_chain_metadata_compatibility([None, None], ["f0", "f1"]) is None

    @pytest.mark.parametrize("mutate", [
        lambda m: m["aliases"].__setitem__("d", "x/z"),
        lambda m: m["column_dtypes"].__setitem__("a", "int32"),
        lambda m: m["subframe_indices"]["R"].__setitem__("index_columns", ["row"]),
    ])
    def test_mismatch_raises(self, mutate):
        b = _real_meta(); mutate(b)
        with pytest.raises(ChainMetadataCompatibilityError):
            AliasDataFrame._check_chain_metadata_compatibility([_real_meta(), b], ["f0", "f1"])

    def test_mixed_presence_raises(self):
        with pytest.raises(ChainMetadataCompatibilityError):
            AliasDataFrame._check_chain_metadata_compatibility([_real_meta(), None], ["f0", "f1"])


class TestApplyRealStructure:
    def test_applies_aliases(self):
        adf = AliasDataFrame(pd.DataFrame({"x": [1., 2.], "y": [3., 4.]}))
        adf._apply_recovered_metadata(AliasDataFrame._normalize_chain_meta(_real_meta()))
        assert "d" in adf.aliases

    def test_names_only_apply_is_noop(self):
        adf = AliasDataFrame(pd.DataFrame({"x": [1.]}))
        n = AliasDataFrame._normalize_chain_meta(
            {"schema_source": "names_only", "subframes": ["X"], "aliases": {}, "column_dtypes": {}})
        before = set(adf.df.columns)
        adf._apply_recovered_metadata(n)
        assert set(adf.df.columns) == before and "d" not in adf.aliases


class TestValidateMetadataParam:
    def test_bad_value_raises(self):
        with pytest.raises(ValueError, match="validate_metadata"):
            AliasDataFrame.read_chain_lazy(["x.root:t"], validate_metadata="typo")

    def test_selector_rejects_bad_value(self):
        # Rev 3.1: validate_metadata is a strict-only selector; only bad values raise here.
        with pytest.raises(ValueError, match="validate_metadata must be"):
            AliasDataFrame.read_chain_lazy(["nonexistent.root:t"], validate_metadata="warn")
    # NOTE: union/intersection SKIP metadata recovery (behavior-preserving); verified on alma2.
