"""Dedicated semantic tests for the vector x facet summary-fit P0.

BUG_dfdraw_20260910_vector_facet_channel_assignment_fit_aggregation, P0-1
(found by GPT15 in the CRR v1.0 review).

THE DEFECT
    With both vector dispatch and facet_by active, each branch's stats['fit']
    is itself a facet-keyed dict. Aggregating it under a (branch,) key nested
    two Shape 3 levels, and the summary-fit flattener then read the OUTER key
    as the facet -- formatting the BRANCH INDEX as a facet value -- and the
    INNER keys as groups. Result: the two dimensions were silently swapped and
    a `group` column appeared although no group_by was ever requested. Fit
    numbers were correct; the labels lied, so a fit was attributed to the
    wrong facet.

WHY THE ORIGINAL TESTS MISSED IT
    They asserted that a summary_fit payload EXISTED. Presence cannot detect
    permuted labels. These tests assert row IDENTITY instead.

THE FIXTURE CONSTRAINT THAT MAKES THIS DETECTABLE
    Every dimension must have a DISTINCT cardinality. With 3 facets and 2
    branches, a facet/branch swap cannot produce a valid labelling: a swapped
    facet axis would show 2 distinct values where 3 exist. With a symmetric
    2 x 2 fixture the swap is INVISIBLE no matter how strong the assertions
    are -- which is exactly how this defect survived a green suite.
    test_fixture_cardinalities_are_asymmetric below locks that property, so a
    later fixture edit cannot quietly disarm the whole file.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dfextensions.dfdraw import DFDraw  # noqa: E402


N_FACETS = 3      # side_bin takes 0, 1, 2
N_BRANCHES = 2    # two selection-vector branches
SEL_VECTOR = ["sector == 0", "sector == 1"]


@pytest.fixture
def df_asym():
    """Asymmetric fixture: 3 facets x 2 branches (see module docstring)."""
    rng = np.random.default_rng(20260911)
    n = 9000
    return pd.DataFrame({
        "x": rng.normal(0.0, 1.0, n),
        "y": rng.normal(0.0, 1.0, n),
        "sector": rng.integers(0, 4, n),
        "side_bin": rng.integers(0, N_FACETS, n),
        "mult_bin": rng.integers(0, 5, n),
    })


def _summary(d, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, stats = d.profile(
            "y:x", facet_by="side_bin",
            selection_vector=SEL_VECTOR, vector_compose="outer",
            fit="pol1", summary_fit="table", **kwargs
        )
    assert isinstance(stats, list), f"expected list, got {type(stats).__name__}"
    return stats


def test_fixture_cardinalities_are_asymmetric(df_asym):
    """Guard: the whole file is only meaningful on asymmetric cardinalities."""
    assert df_asym["side_bin"].nunique() == N_FACETS
    assert N_FACETS != N_BRANCHES, (
        "facet count equals branch count: a facet/branch permutation would be "
        "undetectable and every test in this file becomes vacuous"
    )


def test_summary_fit_is_a_real_payload_not_a_note(df_asym):
    """Strengthened T8/T9: no 'payload OR note' escape hatch.

    The original tests accepted a diagnostic note in place of a payload, so a
    deliberately broken consumer passed them.
    """
    stats = _summary(DFDraw(df_asym))
    assert stats[0].get("summary_fit_note") is None, (
        f"summary_fit degraded to a note: {stats[0].get('summary_fit_note')}"
    )
    payload = stats[0].get("summary_fit")
    assert isinstance(payload, dict) and payload, "no summary_fit payload"
    assert "data" in payload, f"payload has no 'data': {sorted(payload)}"
    assert payload["data"], "summary_fit data is empty"


def test_row_count_is_the_full_facet_by_branch_cross_product(df_asym):
    """A collapsed or duplicated dimension changes the row count."""
    rows = _summary(DFDraw(df_asym))[0]["summary_fit"]["data"]
    assert len(rows) == N_FACETS * N_BRANCHES, (
        f"expected {N_FACETS} facets x {N_BRANCHES} branches = "
        f"{N_FACETS * N_BRANCHES} rows, got {len(rows)}"
    )


def test_every_actual_facet_value_appears_in_the_facet_label(df_asym):
    """THE decisive assertion: the facet axis must carry the real facets.

    Before the fix the facet label enumerated the BRANCH index, so this set
    had N_BRANCHES (2) members instead of N_FACETS (3).
    """
    rows = _summary(DFDraw(df_asym))[0]["summary_fit"]["data"]
    labels = [str(r.get("facet")) for r in rows]
    for facet_value in range(N_FACETS):
        needle = f"side_bin={facet_value}"
        assert any(needle in lab for lab in labels), (
            f"facet value {facet_value} never appears in any summary row; "
            f"labels were {sorted(set(labels))}"
        )


def test_branch_identity_is_present_and_distinct(df_asym):
    """Each facet must appear once per branch, distinguishably."""
    rows = _summary(DFDraw(df_asym))[0]["summary_fit"]["data"]
    labels = [str(r.get("facet")) for r in rows]
    assert len(set(labels)) == N_FACETS * N_BRANCHES, (
        "summary rows are not uniquely identified by (facet, branch); "
        f"{len(set(labels))} distinct labels for {len(rows)} rows: "
        f"{sorted(set(labels))}"
    )
    for facet_value in range(N_FACETS):
        matching = [lab for lab in labels if f"side_bin={facet_value}" in lab]
        assert len(matching) == N_BRANCHES, (
            f"facet {facet_value} has {len(matching)} rows, "
            f"expected {N_BRANCHES} (one per branch)"
        )


def test_group_is_unset_when_no_group_by_requested(df_asym):
    """Before the fix the facet values were reported as groups."""
    rows = _summary(DFDraw(df_asym))[0]["summary_fit"]["data"]
    populated = [r.get("group") for r in rows if r.get("group") is not None]
    assert not populated, (
        "no group_by was requested, but the summary reports groups "
        f"{populated} -- facet values are being mislabelled as groups"
    )


def test_group_by_populates_group_without_corrupting_facet(df_asym):
    """With a real group_by, group fills in and facet stays truthful."""
    stats = _summary(DFDraw(df_asym), group_by="mult_bin", group_by_bins=2)
    rows = stats[0]["summary_fit"]["data"]
    assert rows, "no summary rows with group_by active"
    labels = [str(r.get("facet")) for r in rows]
    for facet_value in range(N_FACETS):
        assert any(f"side_bin={facet_value}" in lab for lab in labels), (
            f"facet {facet_value} lost from the facet axis once group_by was "
            f"added; labels {sorted(set(labels))}"
        )


def test_fit_payload_survives_alongside_correct_labels(df_asym):
    """Labels are right AND the numbers are still there."""
    rows = _summary(DFDraw(df_asym))[0]["summary_fit"]["data"]
    named = [r for r in rows if r.get("fit_name")]
    assert len(named) == len(rows), (
        f"{len(rows) - len(named)} rows carry no fit_name"
    )
    assert all(r.get("fit_name") == "pol1" for r in named), (
        f"unexpected fit names: {sorted({r.get('fit_name') for r in named})}"
    )


def test_plain_vector_summary_fit_unchanged_by_the_fix(df_asym):
    """Negative control: no facet_by -> pre-existing branch-keyed behaviour.

    For plain vector dispatch the established convention keys rows by branch.
    The composite-key path must not disturb that.
    """
    d = DFDraw(df_asym)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, stats = d.profile(
            "y:x", selection_vector=SEL_VECTOR, vector_compose="outer",
            fit="pol1", summary_fit="table",
        )
    assert isinstance(stats, list)
    payload = stats[0].get("summary_fit")
    assert isinstance(payload, dict) and payload.get("data"), (
        "plain vector summary_fit regressed"
    )
    assert stats[0].get("summary_fit_note") is None


def test_ordinary_facet_summary_fit_unchanged_by_the_fix(df_asym):
    """Negative control: no vector -> dict stats, facet labels, group unset."""
    d = DFDraw(df_asym)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, stats = d.profile(
            "y:x", facet_by="side_bin", fit="pol1", summary_fit="table",
        )
    assert isinstance(stats, dict), (
        f"ordinary facet contract broken: {type(stats).__name__}"
    )
    rows = stats.get("summary_fit", {}).get("data")
    assert rows, "ordinary faceted summary_fit regressed"
    labels = [str(r.get("facet")) for r in rows]
    for facet_value in range(N_FACETS):
        assert any(f"side_bin={facet_value}" in lab for lab in labels)
    assert all(r.get("group") is None for r in rows), (
        "ordinary faceted path now reports groups without group_by"
    )
