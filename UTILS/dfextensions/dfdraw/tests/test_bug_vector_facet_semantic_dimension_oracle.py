"""Semantic-dimension oracle for vector x facet x summary-fit.

Implements `GPT17_dfdraw_SemanticDimensionOracle_VectorFacetSummaryFit_Test_Proposal_20260911.md`
(T-SDO-1 .. T-SDO-6 plus the §9 grouped companion), for
BUG_dfdraw_20260910_vector_facet_channel_assignment_fit_aggregation.

THE RULE THIS FILE ENFORCES (proposal §13)
    When a public draw result combines several orthogonal semantic dimensions,
    a test must prove the identity of the full Cartesian coordinate set, and
    attach an independently known numerical observable to each coordinate.
    Shape, non-emptiness, rendering success and deterministic order are NOT
    sufficient. The previous test round asserted only presence and therefore
    let a label swap ship.

WHY THE FIXTURE LOOKS LIKE THIS
    * 3 branches x 2 facets -- unequal cardinalities, so a facet/branch
      permutation cannot produce a valid labelling.
    * unequal branch populations (5 / 7 / 11 repeats) -- so a stable-but-
      swapped branch order fails on row counts alone.
    * unique (slope, intercept) per branch x facet cell -- so a fit payload
      attached to the wrong label fails on the numbers (mutation M7), not just
      on the labels.
    * y values are the exact generated line plus SYMMETRIC offsets that sum to
      exactly zero. Two deviations from the proposal, both necessary:
        - the proposal's zero-scatter fixture makes every per-bin error zero,
          the pol1 fit then reports fit_status='failed' with n_data=0, and no
          slope/intercept columns are produced at all -- so T-SDO-5 cannot be
          built on it. Zero-sum offsets keep the bin MEAN exactly on the
          generated line (so the truth stays exactly known) while giving the
          profile a non-zero error to fit against.
        - the proposal omits vector_compose='outer'; with 3 selections and one
          y expression the call raises
          "3-axis inner requires equal lengths: vector=1, selection_vector=3".
      Both are recorded in the CRR as proposal errata.
    * the facet column is named `facet_val`, not `facet`, so a bug that echoed
      the row-key name instead of the real column value could not pass by
      coincidence.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dfextensions.dfdraw import DFDraw  # noqa: E402


# ---------------------------------------------------------------- truth model
BRANCHES = (0, 1, 2)
BRANCH_REPEATS = {0: 5, 1: 7, 2: 11}      # deliberately unequal
FACET_VALUES = (10, 20)
BIN_CENTERS = (0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5)
EPS = 1e-3


def expected_slope(branch, facet_index):
    return 10.0 * (branch + 1) + (facet_index + 1)


def expected_intercept(branch, facet_index):
    return 100.0 * branch + 3.0 * facet_index


@pytest.fixture
def df_oracle():
    """Exactly-known linear fixture; truth is independent of dfdraw."""
    rows = []
    for branch in BRANCHES:
        k = BRANCH_REPEATS[branch]
        for facet_index, facet in enumerate(FACET_VALUES):
            slope = expected_slope(branch, facet_index)
            intercept = expected_intercept(branch, facet_index)
            for x in BIN_CENTERS:
                for j in range(k):
                    # symmetric offsets summing to exactly zero
                    offset = (j - (k - 1) / 2.0) * EPS
                    rows.append({
                        "branch": branch,
                        "facet_val": facet,
                        "x": x,
                        "y": intercept + slope * x + offset,
                    })
    return pd.DataFrame(rows)


def _run(df, **extra):
    """The real public entry point, not a private helper."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DFDraw(df).profile(
            "y:x", bins=8, range=(0.0, 8.0),
            selection_vector=[f"branch=={b}" for b in BRANCHES],
            selection_labels=[f"B{b}" for b in BRANCHES],
            facet_by="facet_val",
            vector_compose="outer",
            fit="pol1", summary_fit="table", return_data=True,
            **extra
        )


def _payload(stats):
    assert isinstance(stats, list), f"expected list, got {type(stats).__name__}"
    assert stats[0].get("summary_fit_note") is None, (
        f"summary degraded to a note: {stats[0].get('summary_fit_note')}"
    )
    payload = stats[0].get("summary_fit")
    assert isinstance(payload, dict) and payload, "no summary_fit payload"
    assert "data" in payload and payload["data"], "summary_fit data empty"
    return payload


def _coord(row):
    """(facet_value, branch_index) parsed from the public facet label.

    The label is the established Shape 3 display form, e.g.
    ``facet_val=10 branch=2``. Parsed rather than reconstructed so the test
    reads what a user reads.
    """
    label = str(row.get("facet"))
    facet = branch = None
    for token in label.split():
        if token.startswith("facet_val="):
            facet = int(float(token.split("=", 1)[1]))
        elif token.startswith("branch="):
            branch = int(float(token.split("=", 1)[1]))
    return facet, branch


EXPECTED_COORDS = {(f, b) for b in BRANCHES for f in FACET_VALUES}


# --------------------------------------------------------------------------
# T-SDO-1 -- branch identity and order, bound to independent raw counts
# --------------------------------------------------------------------------

def test_SDO_1_branch_identity_bound_to_raw_counts(df_oracle):
    _, _, stats = _run(df_oracle)
    assert len(stats) == len(BRANCHES), (
        f"expected {len(BRANCHES)} branches, got {len(stats)}"
    )
    expected_n = [int((df_oracle["branch"] == b).sum()) for b in BRANCHES]
    assert [s["n_total"] for s in stats] == expected_n, (
        f"branch order/identity wrong: got {[s['n_total'] for s in stats]}, "
        f"expected {expected_n}"
    )
    assert len(set(expected_n)) == len(expected_n), (
        "branch populations are not all distinct; a stable swap would pass"
    )


# --------------------------------------------------------------------------
# T-SDO-2 -- complete branch-level faceted fits, no any() acceptance
# --------------------------------------------------------------------------

def test_SDO_2_every_branch_has_a_fit_for_every_facet(df_oracle):
    _, _, stats = _run(df_oracle)
    expected_keys = {(str(f),) for f in FACET_VALUES}
    for i, branch_stats in enumerate(stats):
        fit = branch_stats.get("fit")
        assert isinstance(fit, dict) and fit, f"branch {i} has no fit"
        assert set(fit.keys()) == expected_keys, (
            f"branch {i} fit facets {sorted(fit.keys())} != "
            f"{sorted(expected_keys)}"
        )


# --------------------------------------------------------------------------
# T-SDO-3 -- successful summary consumption, exact cardinality
# --------------------------------------------------------------------------

def test_SDO_3_summary_payload_real_and_exact_row_count(df_oracle):
    _, _, stats = _run(df_oracle)
    payload = _payload(stats)
    assert len(payload["data"]) == len(EXPECTED_COORDS), (
        f"expected {len(EXPECTED_COORDS)} rows "
        f"({len(BRANCHES)} branches x {len(FACET_VALUES)} facets), "
        f"got {len(payload['data'])}"
    )


# --------------------------------------------------------------------------
# T-SDO-4 -- exact semantic coordinate set (the direct P0 lock)
# --------------------------------------------------------------------------

def test_SDO_4_exact_semantic_coordinate_product(df_oracle):
    _, _, stats = _run(df_oracle)
    rows = _payload(stats)["data"]
    coords = [_coord(r) for r in rows]

    assert None not in [c[0] for c in coords], (
        f"a row carries no real facet value: {[str(r.get('facet')) for r in rows]}"
    )
    assert None not in [c[1] for c in coords], (
        f"a row carries no branch identity: {[str(r.get('facet')) for r in rows]}"
    )
    assert {c[0] for c in coords} == set(FACET_VALUES), (
        f"facet axis wrong: {sorted({c[0] for c in coords})} != "
        f"{sorted(FACET_VALUES)}"
    )
    assert {c[1] for c in coords} == set(BRANCHES), (
        f"branch axis wrong: {sorted({c[1] for c in coords})} != "
        f"{sorted(BRANCHES)}"
    )
    assert set(coords) == EXPECTED_COORDS, (
        f"coordinate set wrong: {sorted(set(coords))} != "
        f"{sorted(EXPECTED_COORDS)}"
    )
    assert len(coords) == len(set(coords)), (
        f"duplicate coordinates: {sorted(coords)}"
    )
    assert all(r.get("group") is None for r in rows), (
        "no group_by requested, yet rows report groups"
    )


# --------------------------------------------------------------------------
# T-SDO-5 -- independently known fit parameters per coordinate (mutation M7)
# --------------------------------------------------------------------------

def test_SDO_5_fit_params_match_generated_truth_per_coordinate(df_oracle):
    """A correct label with the wrong payload must fail here."""
    _, _, stats = _run(df_oracle)
    rows = _payload(stats)["data"]

    checked = 0
    for row in rows:
        facet, branch = _coord(row)
        facet_index = FACET_VALUES.index(facet)
        want_slope = expected_slope(branch, facet_index)
        want_intercept = expected_intercept(branch, facet_index)

        assert row.get("fit_status") == "ok", (
            f"coordinate (facet={facet}, branch={branch}) fit_status="
            f"{row.get('fit_status')} n_data={row.get('n_data')}"
        )
        assert row.get("slope") == pytest.approx(want_slope, abs=1e-3), (
            f"(facet={facet}, branch={branch}) slope {row.get('slope')} != "
            f"expected {want_slope} -- fit payload attached to the wrong "
            f"semantic coordinate"
        )
        assert row.get("intercept") == pytest.approx(want_intercept, abs=1e-2), (
            f"(facet={facet}, branch={branch}) intercept "
            f"{row.get('intercept')} != expected {want_intercept}"
        )
        checked += 1

    assert checked == len(EXPECTED_COORDS), (
        f"only {checked} coordinates carried checkable parameters"
    )


# --------------------------------------------------------------------------
# T-SDO-6 -- the RENDERED table matches the validated rows
# --------------------------------------------------------------------------

def _rendered_identity_cells(payload):
    """Identity-column cells of the rendered matplotlib table."""
    table_fig = payload.get("table")
    assert table_fig is not None, "summary_fit carries no rendered table"
    cells = None
    for ax in table_fig.axes:
        for tbl in getattr(ax, "tables", []):
            cells = tbl.get_celld()
            break
        if cells:
            break
    assert cells, "no matplotlib Table found in the rendered summary figure"

    n_rows = max(r for r, _ in cells) + 1
    header = str(cells[(0, 0)].get_text().get_text())
    assert header == "facet", (
        f"identity column is not the first column (header={header!r})"
    )
    return {
        str(cells[(r, 0)].get_text().get_text())
        for r in range(1, n_rows)
    }


def test_SDO_6_rendered_table_identities_match_validated_rows(df_oracle):
    """The user-visible table must not permute or relabel validated data."""
    _, _, stats = _run(df_oracle)
    payload = _payload(stats)

    from_data = {str(r.get("facet")) for r in payload["data"]}
    from_render = _rendered_identity_cells(payload)

    assert from_render == from_data, (
        "rendered table identity cells differ from the validated summary "
        f"rows.\n rendered: {sorted(from_render)}\n data:     {sorted(from_data)}"
    )
    assert len(from_render) == len(EXPECTED_COORDS), (
        f"rendered table has {len(from_render)} identity rows, expected "
        f"{len(EXPECTED_COORDS)}"
    )


# --------------------------------------------------------------------------
# §9 grouped companion -- branch x facet x group must stay three dimensions
# --------------------------------------------------------------------------

@pytest.fixture
def df_grouped():
    """2 branches x 2 facets x 2 groups = 8 semantic cells."""
    rows = []
    repeats = {0: 5, 1: 7}
    for branch in (0, 1):
        k = repeats[branch]
        for facet_index, facet in enumerate(FACET_VALUES):
            for group in (0, 1):
                slope = 10.0 * (branch + 1) + (facet_index + 1) + 0.5 * group
                intercept = 100.0 * branch + 3.0 * facet_index + 7.0 * group
                for x in BIN_CENTERS:
                    for j in range(k):
                        offset = (j - (k - 1) / 2.0) * EPS
                        rows.append({
                            "branch": branch,
                            "facet_val": facet,
                            "grp": group,
                            "x": x,
                            "y": intercept + slope * x + offset,
                        })
    return pd.DataFrame(rows)


def test_SDO_9_grouped_keeps_three_dimensions_separate(df_grouped):
    """Guards against fixing the no-group case by special-casing group=None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, stats = DFDraw(df_grouped).profile(
            "y:x", bins=8, range=(0.0, 8.0),
            selection_vector=["branch==0", "branch==1"],
            selection_labels=["B0", "B1"],
            facet_by="facet_val", group_by="grp",
            vector_compose="outer",
            fit="pol1", summary_fit="table", return_data=True,
        )
    payload = _payload(stats)
    rows = payload["data"]

    # facet axis must still carry the real facet values
    facets = {_coord(r)[0] for r in rows}
    assert facets == set(FACET_VALUES), (
        f"facet axis corrupted once group_by was added: {sorted(facets)}"
    )
    # branch axis must still be present and complete
    branches = {_coord(r)[1] for r in rows}
    assert branches == {0, 1}, f"branch axis wrong: {sorted(branches)}"
    # group axis must now be populated, and must NOT be the facet values
    groups = {r.get("group") for r in rows}
    assert groups and groups != {None}, (
        "group_by was requested but no row reports a group"
    )
    assert not (groups & set(FACET_VALUES)), (
        f"facet values are leaking into the group axis: {sorted(groups)}"
    )
