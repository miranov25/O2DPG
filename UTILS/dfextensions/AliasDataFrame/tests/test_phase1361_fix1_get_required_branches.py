"""
Phase 13.61.ADF — Fix-1 (A1 first commit): get_required_branches must route
group_by and color through _add_colname_kwarg (expression analysis), not add them
literally. AC-1b gate.

FM#12: these tests call the same public API the draw-path projection will use
(get_required_branches) and FAIL against pre-fix code:
  pre-fix : get_required_branches(group_by="abs(qpt)") -> {"abs(qpt)"}
  post-fix: get_required_branches(group_by="abs(qpt)") -> {"qpt"}
"""
import numpy as np
import pandas as pd
import pytest
from AliasDataFrame import AliasDataFrame


@pytest.fixture
def adf():
    # Eager ADF; qpt/tgl are real columns, x/y decoys for expr context.
    df = pd.DataFrame({
        "qpt": np.random.randn(200),
        "tgl": np.random.randn(200),
        "x": np.random.randn(200),
        "y": np.random.randn(200),
    })
    return AliasDataFrame(df)


def _branches(adf, **kw):
    return set(adf.get_required_branches(**kw))


@pytest.mark.invariance
def test_AC1b_group_by_expression_contributes_column_not_literal(adf):
    # Load-bearing (FM#12): pre-fix returns {"abs(qpt)"}; post-fix returns {"qpt"}.
    assert _branches(adf, group_by="abs(qpt)") == {"qpt"}


def test_group_by_literal_column_unchanged(adf):
    assert _branches(adf, group_by="tgl") == {"tgl"}


def test_color_literal_filtered_by_validate(adf):
    # R4-C3: a literal matplotlib color must not become a phantom projection ref
    # when the eager projection call uses validate=True.
    branches = _branches(adf, expr="y:x", color="red", validate=True)
    assert "red" not in branches


def test_color_expression_contributes_refs(adf):
    assert _branches(adf, color="abs(qpt)") == {"qpt"}


@pytest.mark.invariance
def test_facet_by_expression_unchanged_regression(adf):
    # facet_by already routed through _add_colname_kwarg; must be unaffected.
    assert _branches(adf, facet_by="abs(tgl)") == {"tgl"}


def test_group_by_equals_facet_by_for_same_expression(adf):
    # Symmetry: group_by and facet_by resolve an expression identically.
    assert _branches(adf, group_by="abs(qpt)") == _branches(adf, facet_by="abs(qpt)")
