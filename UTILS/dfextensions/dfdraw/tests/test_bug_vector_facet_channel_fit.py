"""T1-T10 for BUG_dfdraw_20260910_vector_facet_channel_assignment_fit_aggregation.

Two defects are covered, both exposed by ADF PHASE_13_76_ADF B3.3e:

  BUG-A  channel assignment was recomputed once per facet panel, so
         assign_channels() ran N times for one logical plot. That breaks the
         "at most once per logical plot" invariant and, more importantly,
         allows the same vector branch to receive a different colour /
         linestyle / marker in different panels.

  BUG-B  vector-aware faceted stats became list[dict], and the old dict-only
         fit aggregation was skipped, so the branch-level 'fit' was absent and
         summary_fit could not consume it.

Test numbering follows the bug report:

  T1   assign_channels called at most once (the existing IAP-5 invariant,
       re-asserted here at bug scope)
  T2   selection_vector x facet_by -> vector-major list, order, keys
  T3   weights_vector x facet_by -> same
  T4   ordinary facet still returns dict, and its 'fit' still lands at
       stats['fit'] (negative control for the BUG-B branch)
  T5   cross-panel visual-channel consistency, compared against the
       un-faceted resolution (the semantic falsifier behind T1)
  T6   selection_vector x facet_by x fit -> branch-level 'fit' present
  T7   weights_vector x facet_by x fit -> same
  T8   selection_vector x facet_by x fit x summary_fit -> consumable
  T9   weights_vector x facet_by x fit x summary_fit -> same
  T10  explicit range= survives the x_range -> range translation
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import dfextensions.dfdraw.channels as _channels_mod  # noqa: E402
from dfextensions.dfdraw import DFDraw  # noqa: E402
from dfextensions.dfdraw.channels import assign_channels  # noqa: E402


@pytest.fixture
def df_vec():
    rng = np.random.default_rng(20260910)
    n = 6000
    return pd.DataFrame({
        "x": rng.normal(0.0, 1.0, n),
        "y": rng.normal(0.0, 1.0, n),
        "w1": rng.uniform(0.5, 1.5, n),
        "w2": rng.uniform(0.5, 1.5, n),
        "sector": rng.integers(0, 4, n),
        "quartile_val": rng.integers(0, 4, n),
    })


SEL_VECTOR = ["sector == 0", "sector == 1"]
W_VECTOR = ["w1", "w2"]


def _draw(d, **kwargs):
    """Run a draw with warnings silenced; return (fig, ax, stats)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return d.profile("y:x", **kwargs)


# --------------------------------------------------------------------------
# T1 -- assign_channels at most once per logical plot
# --------------------------------------------------------------------------

def test_T1_assign_channels_called_at_most_once_vector_facet(df_vec):
    """BUG-A: one logical plot resolves channels once, not once per panel."""
    d = DFDraw(df_vec)
    with patch(
        "dfextensions.dfdraw.channels.assign_channels", wraps=assign_channels
    ) as spy:
        _draw(d, facet_by="quartile_val",
              selection_vector=SEL_VECTOR, vector_compose="outer")
    assert spy.call_count <= 1, (
        f"assign_channels should be called at most once (got {spy.call_count}); "
        "channel resolution is leaking into the per-facet path"
    )


def test_T1b_assign_channels_at_most_once_weights_vector(df_vec):
    d = DFDraw(df_vec)
    with patch(
        "dfextensions.dfdraw.channels.assign_channels", wraps=assign_channels
    ) as spy:
        _draw(d, facet_by="quartile_val",
              weights_vector=W_VECTOR, vector_compose="outer")
    assert spy.call_count <= 1, f"got {spy.call_count}"


# --------------------------------------------------------------------------
# T2 / T3 -- vector-major return shape and order
# --------------------------------------------------------------------------

_FACET_KEYS = ("n_groups", "groups", "per_group", "faceted",
               "facet_by", "facet_mode", "n_total")


def _assert_vector_major(stats, n_branches, facet_by="quartile_val"):
    assert isinstance(stats, list), f"expected list, got {type(stats).__name__}"
    assert len(stats) == n_branches, f"expected {n_branches} branches"
    for i, branch in enumerate(stats):
        assert isinstance(branch, dict), f"branch {i} is {type(branch).__name__}"
        for key in _FACET_KEYS:
            assert key in branch, f"branch {i} missing {key!r}"
        assert branch["faceted"] is True
        assert branch["facet_by"] == facet_by
        assert isinstance(branch["per_group"], dict)
        assert len(branch["per_group"]) == branch["n_groups"]


def test_T2_selection_vector_facet_returns_vector_major_list(df_vec):
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val",
                        selection_vector=SEL_VECTOR, vector_compose="outer")
    _assert_vector_major(stats, n_branches=len(SEL_VECTOR))


def test_T2b_selection_branch_order_matches_selection_order(df_vec):
    """Branch order must follow selection order, not dict iteration luck."""
    d = DFDraw(df_vec)
    _, _, first = _draw(d, facet_by="quartile_val",
                        selection_vector=SEL_VECTOR, vector_compose="outer")
    _, _, second = _draw(d, facet_by="quartile_val",
                         selection_vector=SEL_VECTOR, vector_compose="outer")
    assert [b["n_total"] for b in first] == [b["n_total"] for b in second]
    # B3 strengthening (CRR review P1-2): repeatability alone would accept a
    # stable-but-SWAPPED order. Bind each returned branch index to an
    # independently computed raw-input count instead.
    expected = [
        int((df_vec["sector"] == 0).sum()),
        int((df_vec["sector"] == 1).sum()),
    ]
    assert [b["n_total"] for b in first] == expected, (
        f"branch order does not match selection order: got "
        f"{[b['n_total'] for b in first]}, expected {expected}"
    )
    assert expected[0] != expected[1], (
        "fixture no longer distinguishes the branches; test is vacuous"
    )


def test_T3_weights_vector_facet_returns_vector_major_list(df_vec):
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val",
                        weights_vector=W_VECTOR, vector_compose="outer")
    _assert_vector_major(stats, n_branches=len(W_VECTOR))


# --------------------------------------------------------------------------
# T4 -- ordinary facet unchanged (negative control for the BUG-B branch)
# --------------------------------------------------------------------------

def test_T4_ordinary_facet_still_returns_dict(df_vec):
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val")
    assert isinstance(stats, dict), (
        f"ordinary facet contract broken: got {type(stats).__name__}"
    )
    for key in _FACET_KEYS:
        assert key in stats


def test_T4b_ordinary_facet_fit_still_lands_at_top_level(df_vec):
    """The BUG-B fix must not break the dict fit path it branches away from."""
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val", fit="pol1")
    assert isinstance(stats, dict)
    assert "fit" in stats, "ordinary faceted fit aggregation regressed"
    assert isinstance(stats["fit"], dict) and stats["fit"], "fit payload empty"


# --------------------------------------------------------------------------
# T5 -- cross-panel visual-channel consistency (the reason behind T1)
# --------------------------------------------------------------------------

def test_T5_resolved_assignment_matches_unfaceted_resolution(df_vec):
    """Same branch -> same visual channel, and the same as without faceting.

    Panels agreeing with each other but disagreeing with the un-faceted
    resolution would still pass a panel-to-panel comparison, so compare
    against the un-faceted assignment.
    """
    d = DFDraw(df_vec)

    faceted = []
    with patch(
        "dfextensions.dfdraw.channels.assign_channels", wraps=assign_channels
    ) as spy:
        _draw(d, facet_by="quartile_val",
              selection_vector=SEL_VECTOR, vector_compose="outer")
        faceted = [c for c in spy.call_args_list]

    plain = []
    with patch(
        "dfextensions.dfdraw.channels.assign_channels", wraps=assign_channels
    ) as spy:
        _draw(d, selection_vector=SEL_VECTOR, vector_compose="outer")
        plain = [c for c in spy.call_args_list]

    assert len(faceted) == 1 and len(plain) == 1, (
        f"expected one resolution each, got faceted={len(faceted)} "
        f"plain={len(plain)}"
    )
    # Same DataChannel inputs -> Algorithm A yields the same assignment.
    faceted_names = [ch.name for ch in faceted[0].args[0]]
    plain_names = [ch.name for ch in plain[0].args[0]]
    assert faceted_names == plain_names, (
        f"faceted resolution saw different channels: {faceted_names} "
        f"vs un-faceted {plain_names}"
    )
    assert assign_channels(faceted[0].args[0]) == assign_channels(plain[0].args[0]), (
        "faceted vector plot resolves a different branch -> channel mapping "
        "than the same un-faceted call"
    )


def test_T5b_line_styles_identical_across_panels(df_vec):
    """Rendered artists: each branch keeps one style in every panel."""
    d = DFDraw(df_vec)
    fig, axes, stats = _draw(d, facet_by="quartile_val",
                             selection_vector=SEL_VECTOR,
                             vector_compose="outer")
    axes_flat = np.atleast_1d(axes).ravel()
    per_panel = []
    for ax in axes_flat:
        lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0]
        if len(lines) >= len(SEL_VECTOR):
            per_panel.append([
                (ln.get_color(), ln.get_linestyle(), ln.get_marker())
                for ln in lines[:len(SEL_VECTOR)]
            ])
    assert len(per_panel) >= 2, "need >=2 populated panels to compare"
    for panel in per_panel[1:]:
        assert panel == per_panel[0], (
            f"branch styles differ between panels: {per_panel[0]} vs {panel}"
        )


# --------------------------------------------------------------------------
# T6 / T7 -- branch-level fit is present, not silently dropped
# --------------------------------------------------------------------------

def test_T6_selection_vector_facet_fit_present_per_branch(df_vec):
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val",
                        selection_vector=SEL_VECTOR, vector_compose="outer",
                        fit="pol1")
    # B4 strengthening (CRR review P1-3): require fits on EVERY branch and
    # EVERY facet, not merely on at least one.
    assert isinstance(stats, list) and len(stats) == len(SEL_VECTOR)
    expected_keys = {(str(g),) for g in range(4)}
    for i, branch in enumerate(stats):
        fit = branch.get("fit")
        assert isinstance(fit, dict) and fit, (
            f"BUG-B: branch {i} carries no 'fit' payload"
        )
        assert set(fit.keys()) == expected_keys, (
            f"branch {i} fit keys {sorted(fit.keys())} != {sorted(expected_keys)}"
        )


def test_T7_weights_vector_facet_fit_present_per_branch(df_vec):
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val",
                        weights_vector=W_VECTOR, vector_compose="outer",
                        fit="pol1")
    assert isinstance(stats, list) and len(stats) == len(W_VECTOR)
    expected_keys = {(str(g),) for g in range(4)}
    for i, branch in enumerate(stats):
        fit = branch.get("fit")
        assert isinstance(fit, dict) and fit, (
            f"BUG-B: weights branch {i} carries no 'fit' payload"
        )
        assert set(fit.keys()) == expected_keys, (
            f"weights branch {i} fit keys {sorted(fit.keys())}"
        )


# --------------------------------------------------------------------------
# T8 / T9 -- summary_fit can consume the branch-level fits
# --------------------------------------------------------------------------

def _summary_fit_payload(stats):
    """Locate the vector-dispatch summary_fit.

    Documented vector-dispatch behaviour: the rendered summary_fit is written
    into the FIRST iteration dict (stats[0]['summary_fit']) because there is no
    top level when stats is a list. Assert that location explicitly rather than
    searching, so the test cannot pass vacuously.
    """
    assert isinstance(stats, list), f"expected list, got {type(stats).__name__}"
    # B2 strengthening (CRR review P1-1): the original form accepted a
    # diagnostic note INSTEAD of a payload, so a broken consumer passed.
    # Require the note to be absent and the payload to be real.
    note = stats[0].get("summary_fit_note")
    assert note is None, f"summary_fit degraded to a note: {note}"
    payload = stats[0].get("summary_fit")
    assert isinstance(payload, dict) and payload, "no summary_fit payload"
    assert "data" in payload and payload["data"], "summary_fit data empty"
    return payload


def test_T8_selection_vector_facet_fit_summary_fit_consumable(df_vec):
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val",
                        selection_vector=SEL_VECTOR, vector_compose="outer",
                        fit="pol1", summary_fit="table")
    payload = _summary_fit_payload(stats)
    # Full cross-product: 4 facets x 2 branches.
    assert len(payload["data"]) == 4 * len(SEL_VECTOR), (
        f"expected {4 * len(SEL_VECTOR)} summary rows, got {len(payload['data'])}"
    )
    assert all(r.get("group") is None for r in payload["data"]), (
        "no group_by requested, but summary rows report groups"
    )


def test_T9_weights_vector_facet_fit_summary_fit_consumable(df_vec):
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val",
                        weights_vector=W_VECTOR, vector_compose="outer",
                        fit="pol1", summary_fit="table")
    payload = _summary_fit_payload(stats)
    # Full cross-product: 4 facets x 2 branches.
    assert len(payload["data"]) == 4 * len(W_VECTOR), (
        f"expected {4 * len(W_VECTOR)} summary rows, got {len(payload['data'])}"
    )
    assert all(r.get("group") is None for r in payload["data"]), (
        "no group_by requested, but summary rows report groups"
    )


# --------------------------------------------------------------------------
# T10 -- explicit range survives the x_range -> range translation
# --------------------------------------------------------------------------

def test_T10_explicit_range_respected_in_vector_facet(df_vec):
    """The delegation path translates low-level x_range to public range=."""
    d = DFDraw(df_vec)
    lo, hi = -1.5, 1.5
    fig, axes, stats = _draw(d, facet_by="quartile_val",
                             selection_vector=SEL_VECTOR,
                             vector_compose="outer",
                             range=(lo, hi))
    assert isinstance(stats, list)
    # B5 strengthening (CRR review P1-4): the original loop skipped every cell
    # whose range diagnostic was absent, so it could pass with no evidence at
    # all. Require at least one checked cell.
    checked = 0
    for branch in stats:
        for facet_key, cell in branch["per_group"].items():
            if not isinstance(cell, dict):
                continue
            used = cell.get("autorange_used")
            if used is None:
                continue
            checked += 1
            assert used == pytest.approx((lo, hi)), (
                f"facet {facet_key}: explicit range not honoured, got {used}"
            )
    assert checked > 0, (
        "no cell exposed a range diagnostic; the explicit-range assertion "
        "never ran and this test proves nothing"
    )


def test_T10b_no_range_still_autoranges(df_vec):
    """Negative control: without range=, autorange is still engaged."""
    d = DFDraw(df_vec)
    _, _, stats = _draw(d, facet_by="quartile_val",
                        selection_vector=SEL_VECTOR, vector_compose="outer")
    ranges = [
        cell.get("autorange_used")
        for branch in stats
        for cell in branch["per_group"].values()
        if isinstance(cell, dict) and cell.get("autorange_used") is not None
    ]
    assert ranges, (
        "no range diagnostics collected; this negative control proves nothing"
    )
    assert any(r != (-1.5, 1.5) for r in ranges), (
        "autorange appears pinned to the T10 explicit range"
    )
