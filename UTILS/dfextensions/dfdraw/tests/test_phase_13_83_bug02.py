"""PHASE_13_83_DF — BUG-02 pre-fix RED reproducer.

Purpose
-------
Prove the remaining dfdraw defect *before* any BUG-02 product-code change:

    scalar hist + selection_vector=[s0, s1] + column facet
    + vector_compose="outer"

must preserve both semantic dimensions (branch × facet) and must match an
independent NumPy histogram in every cell.

DT-83-1 / AD-62 is also locked: with the default ``inner`` composition, the
same scalar-hist/two-selection request must refuse loudly rather than silently
dropping the vector.

The fixture uses unequal semantic cardinalities (3 facets × 2 branches) so a
branch/facet permutation fails structurally as well as numerically.

This file deliberately includes two passing controls so a RED result cannot be
explained by a broken fixture or artist extractor:
  * non-faceted hist + selection_vector + outer
  * facet-only hist without a vector

Expected on the pre-fix product:
    2 failed, 2 passed

After BUG-02 is fixed:
    4 passed
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfdraw import DFDraw


EDGES = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=float)
N_BINS = len(EDGES) - 1


def _fixture():
    """Asymmetric/correlated fixture with unique truth in every branch×facet cell."""
    # (facet, branch) -> counts in the four explicit histogram bins.
    # Unequal semantic cardinalities are intentional: 3 facets × 2 branches.
    # This makes a branch/facet permutation fail structurally as well as
    # numerically (PHASE_13_81 AC-9 / BUG-02 RED review strengthening).
    cell_counts = {
        (0, 0): [3, 7, 2, 1],
        (0, 1): [1, 2, 9, 4],
        (1, 0): [6, 1, 4, 2],
        (1, 1): [2, 5, 1, 11],
        (2, 0): [4, 8, 3, 6],
        (2, 1): [10, 3, 5, 2],
    }
    centers = np.array([-1.5, -0.5, 0.5, 1.5], dtype=float)

    rows = []
    for (facet, branch), counts in cell_counts.items():
        for x, count in zip(centers, counts):
            rows.extend(
                {"x": float(x), "facet": int(facet), "branch": int(branch)}
                for _ in range(count)
            )

    frame = pd.DataFrame(rows)
    return frame, cell_counts


def _axis_for_facet(fig, facet):
    """Find one visible data axis by the public facet title."""
    token = f"facet={facet}"
    matches = [
        ax
        for ax in fig.axes
        if ax.get_visible() and token in (ax.get_title() or "")
    ]
    assert len(matches) == 1, (
        f"expected exactly one axis titled with {token!r}; "
        f"titles={[ax.get_title() for ax in fig.axes]}"
    )
    return matches[0]


def _bar_series(ax, *, n_bins):
    """Return bar heights as [series][bin], using only rendered Rectangle artists.

    Each ungrouped ``histtype='bar'`` draw contributes exactly ``n_bins``
    Rectangle patches in bin order.  Consecutive vector branches therefore
    contribute consecutive chunks.  This reads only the final artist tree; it
    does not consult dfdraw bookkeeping or intermediate histogram claims.
    """
    patches = list(ax.patches)
    assert len(patches) % n_bins == 0, (
        f"histogram artist exposes {len(patches)} Rectangle patches, "
        f"not an integer number of {n_bins}-bin series"
    )
    return np.asarray(
        [
            [patch.get_height() for patch in patches[i : i + n_bins]]
            for i in range(0, len(patches), n_bins)
        ],
        dtype=float,
    )


def _raw_hist(frame, *, facet=None, branch=None):
    """Independent NumPy truth; no dfdraw primitive or returned stats are used."""
    mask = np.ones(len(frame), dtype=bool)
    if facet is not None:
        mask &= frame["facet"].to_numpy() == facet
    if branch is not None:
        mask &= frame["branch"].to_numpy() == branch
    return np.histogram(frame.loc[mask, "x"].to_numpy(), bins=EDGES)[0].astype(float)


def test_B02_RED_outer_hist_selection_vector_facet_matches_independent_numpy():
    """B02-A/B/C: explicit outer must render two branches in all three facets."""
    frame, cell_counts = _fixture()
    d = DFDraw(frame)

    # Sanity-lock the hand-designed fixture before asking dfdraw anything.
    for key, expected in cell_counts.items():
        np.testing.assert_array_equal(
            _raw_hist(frame, facet=key[0], branch=key[1]),
            np.asarray(expected, dtype=float),
        )

    try:
        fig, _, _ = d.hist(
            "x",
            bins=EDGES,
            histtype="bar",
            selection_vector=["branch == 0", "branch == 1"],
            vector_compose="outer",
            facet_by="facet",
            auto_title=False,
        )

        for facet in (0, 1, 2):
            ax = _axis_for_facet(fig, facet)
            rendered = _bar_series(ax, n_bins=N_BINS)

            # Semantic/cardinality assertion: two requested branches survive.
            assert rendered.shape == (2, N_BINS), (
                f"facet={facet}: expected 2 branch histograms × {N_BINS} bins; "
                f"rendered shape={rendered.shape}"
            )

            # Content-grade oracle: exact raw NumPy bin contents, branch order
            # bound to selection_vector order.
            expected = np.vstack(
                [
                    _raw_hist(frame, facet=facet, branch=0),
                    _raw_hist(frame, facet=facet, branch=1),
                ]
            )
            np.testing.assert_array_equal(rendered, expected)
    finally:
        plt.close("all")


def test_B02_RED_default_inner_refuses_instead_of_silently_dropping_vector():
    """B02-F / DT-83-1 Option A: default inner must refuse scalar×2 selection."""
    frame, _ = _fixture()
    d = DFDraw(frame)

    try:
        with pytest.raises(
            ValueError,
            match=r"(?i)(inner|cardinal|selection_vector|vector)",
        ):
            d.hist(
                "x",
                bins=EDGES,
                histtype="bar",
                selection_vector=["branch == 0", "branch == 1"],
                facet_by="facet",
                auto_title=False,
            )
    finally:
        plt.close("all")


def test_B02_CONTROL_nonfaceted_outer_vector_hist_is_already_correct():
    """B02-G positive control: existing non-faceted histogram vector owner works."""
    frame, _ = _fixture()
    d = DFDraw(frame)

    try:
        fig, ax, _ = d.hist(
            "x",
            bins=EDGES,
            histtype="bar",
            selection_vector=["branch == 0", "branch == 1"],
            vector_compose="outer",
            auto_title=False,
        )
        rendered = _bar_series(ax, n_bins=N_BINS)
        expected = np.vstack(
            [
                _raw_hist(frame, branch=0),
                _raw_hist(frame, branch=1),
            ]
        )
        assert rendered.shape == (2, N_BINS)
        np.testing.assert_array_equal(rendered, expected)
    finally:
        plt.close("all")


def test_B02_CONTROL_facet_only_hist_is_already_correct():
    """B02-I positive control: ordinary hist×facet must remain unchanged."""
    frame, _ = _fixture()
    d = DFDraw(frame)

    try:
        fig, _, _ = d.hist(
            "x",
            bins=EDGES,
            histtype="bar",
            facet_by="facet",
            auto_title=False,
        )
        for facet in (0, 1, 2):
            ax = _axis_for_facet(fig, facet)
            rendered = _bar_series(ax, n_bins=N_BINS)
            assert rendered.shape == (1, N_BINS)
            np.testing.assert_array_equal(
                rendered[0],
                _raw_hist(frame, facet=facet),
            )
    finally:
        plt.close("all")
