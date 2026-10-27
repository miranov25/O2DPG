"""PHASE_13_86_DF Stage-A corrected RED characterization tests.

These tests intentionally characterize current facet-boundary forwarding and
safety defects before any product-code change.  They are content-level tests:
rendering "something" is not sufficient evidence for the requested semantic.

Proposal: PHASE_13_86_DF_FourFacetVectorBugFixes_Proposal_v0_2_20261015.md
Review correction gate:
  PHASE_13_86_DF_StageA_RED_Review_Summary_GPT18_dfdraw_20261015.md

Stage: A — facet forwarding safety and silent-wrong-result stop.
This corrected revision incorporates RED-R1..RED-R4 from the main-reviewer
summary.  Product source must remain unchanged until this file is independently
confirmed.
"""

import inspect
import itertools
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.container as mcontainer
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import pytest

from dfextensions.dfdraw import DFDraw


def _stage_a_df():
    """Small asymmetric fixture: every facet/selection has distinct rows."""
    n_per_facet = 24
    n = 2 * n_per_facet
    facet = np.repeat(["A", "B"], n_per_facet)
    local = np.tile(np.arange(n_per_facet), 2)
    return pd.DataFrame({
        "row_id": np.arange(n, dtype=int),
        "x": np.arange(n, dtype=float),
        "y": 0.25 * np.arange(n, dtype=float) + np.where(facet == "B", 10.0, 0.0),
        "f": facet,
        "g": np.tile(np.repeat(["g0", "g1", "g2"], 8), 2),
        "ex": 0.05 + 0.002 * np.arange(n),
        "ey": 0.10 + 0.003 * np.arange(n),
        "time_s": 1_700_000_000.0 + 1800.0 * np.arange(n),
        "c": np.linspace(-2.0, 3.0, n),
        "sel_left": local < 8,
        "sel_mid": (local >= 8) & (local < 16),
        "w0": np.ones(n),
        "w1": 1.0 + 0.01 * np.arange(n),
    })


def _facet_axes(fig):
    """Return facet data axes, excluding colorbar/helper axes."""
    return [ax for ax in fig.axes if ax.get_title().startswith("f=")]


def _facet_value(ax):
    title = ax.get_title()
    assert title.startswith("f="), title
    return title.split("=", 1)[1]


def _errorbars(ax):
    return [c for c in ax.containers if isinstance(c, mcontainer.ErrorbarContainer)]


def _path_collections(ax):
    return [c for c in ax.collections if isinstance(c, mcollections.PathCollection)]


def _offset_array(collection):
    return np.asarray(collection.get_offsets(), dtype=float)


def _sort_rows_xy(a):
    a = np.asarray(a, dtype=float)
    if len(a) == 0:
        return a.reshape(0, 2)
    order = np.lexsort((a[:, 1], a[:, 0]))
    return a[order]


def _assert_loud_refusal(exc):
    text = str(exc).lower()
    assert "facet" in text, f"Refusal should identify facet composition: {exc}"
    assert any(word in text for word in ("selection", "weight", "vector", "scatter", "unsupported")), (
        f"Refusal should identify the unsupported vector/scatter semantic: {exc}"
    )


class TestStageAPropertyReds:
    """Property-specific REDs required by Proposal v0.2 Stage A1/A2/A2.1."""

    @pytest.mark.parametrize(
        "error_kw, expected_attr",
        [("xerr", "has_xerr"), ("yerr", "has_yerr")],
    )
    def test_A1_scatter_facet_preserves_real_errorbars(self, error_kw, expected_attr):
        df = _stage_a_df()
        fig, _, _ = DFDraw(df).scatter(
            "y:x",
            facet_by="f",
            **{error_kw: "ex" if error_kw == "xerr" else "ey"},
        )
        axes = _facet_axes(fig)
        assert len(axes) == 2
        for ax in axes:
            bars = _errorbars(ax)
            assert len(bars) == 1, (
                f"{ax.get_title()} requested {error_kw} but rendered "
                f"{len(bars)} ErrorbarContainer(s)"
            )
            assert getattr(bars[0], expected_attr), f"{ax.get_title()} missing {error_kw}"
        plt.close(fig)

    @pytest.mark.parametrize("method, expr, extra", [
        ("scatter", "y:time_s", {}),
        ("profile", "y:time_s", {"bins": 6}),
        ("hist", "time_s", {"bins": 6}),
    ])
    def test_A2_faceted_time_format_reaches_each_cell(self, method, expr, extra):
        df = _stage_a_df()
        fig, _, _ = getattr(DFDraw(df), method)(
            expr, facet_by="f", time_format="auto", **extra
        )
        axes = _facet_axes(fig)
        assert len(axes) == 2
        for ax in axes:
            formatter = ax.xaxis.get_major_formatter()
            assert isinstance(formatter, (mdates.AutoDateFormatter, mdates.DateFormatter)), (
                f"{method} {ax.get_title()} dropped time_format; "
                f"got {type(formatter).__name__}"
            )
        plt.close(fig)

    def test_A3_hist_facet_preserves_hist_errors(self):
        df = _stage_a_df()
        fig, _, _ = DFDraw(df).hist("x", bins=6, facet_by="f", hist_errors=True)
        axes = _facet_axes(fig)
        assert len(axes) == 2
        for ax in axes:
            bars = _errorbars(ax)
            assert bars, f"{ax.get_title()} dropped hist_errors=True"
        plt.close(fig)

    def test_A4_profile_group_facet_preserves_explicit_color(self):
        df = _stage_a_df()
        requested = to_rgba("magenta")
        fig, _, _ = DFDraw(df).profile(
            "y:x", bins=6, group_by="g", facet_by="f", color="magenta"
        )
        axes = _facet_axes(fig)
        assert len(axes) == 2
        for ax in axes:
            bars = _errorbars(ax)
            assert len(bars) == 3, f"Expected 3 grouped profile series in {ax.get_title()}"
            central_colors = [to_rgba(c.lines[0].get_color()) for c in bars]
            assert all(np.allclose(color, requested) for color in central_colors), (
                f"{ax.get_title()} dropped explicit color='magenta': {central_colors}"
            )
        plt.close(fig)

    def test_A5_scatter_selection_vector_facet_is_exact_or_refuses(self):
        """RED-R1: exact row membership, not only branch cardinality.

        Supported behavior must render exactly the independent expected
        (x,y) rows for each facet × selection branch.  Branch order is not
        assumed.  If artists expose the supplied branch label, that label must
        identify the matching expected branch.  Loud refusal is also accepted.
        """
        df = _stage_a_df()
        selections = {
            "left": "sel_left",
            "mid": "sel_mid",
        }
        try:
            fig, _, _ = DFDraw(df).scatter(
                "y:x",
                facet_by="f",
                selection_vector=[selections["left"], selections["mid"]],
                selection_labels=["left", "mid"],
                vector_compose="outer",
            )
        except (ValueError, NotImplementedError) as exc:
            _assert_loud_refusal(exc)
            return

        axes = _facet_axes(fig)
        assert len(axes) == 2
        for ax in axes:
            facet_value = _facet_value(ax)
            facet_df = df[df["f"] == facet_value]

            expected_by_label = {}
            for label, sel_col in selections.items():
                branch = facet_df[facet_df[sel_col]]
                expected_by_label[label] = _sort_rows_xy(branch[["x", "y"]].to_numpy())

            collections = _path_collections(ax)
            assert len(collections) == 2, (
                f"{ax.get_title()} expected two selection branches, got "
                f"{len(collections)} PathCollection(s)"
            )

            observed = [_sort_rows_xy(_offset_array(c)) for c in collections]
            assert sorted(map(len, observed)) == [8, 8], (
                f"{ax.get_title()} expected two 8-row branches, got "
                f"{sorted(map(len, observed))}"
            )

            # Exact membership as an unordered two-branch bijection.
            unmatched = list(expected_by_label)
            matched_label_by_index = {}
            for i, offsets in enumerate(observed):
                matches = [
                    label for label in unmatched
                    if offsets.shape == expected_by_label[label].shape
                    and np.allclose(offsets, expected_by_label[label], rtol=0, atol=0)
                ]
                assert len(matches) == 1, (
                    f"{ax.get_title()} collection {i} does not match exactly one "
                    f"expected selection branch; candidates={matches}, "
                    f"offsets={offsets.tolist()}"
                )
                label = matches[0]
                unmatched.remove(label)
                matched_label_by_index[i] = label
            assert not unmatched, f"{ax.get_title()} missing expected branches: {unmatched}"

            # If the renderer exposes branch labels, they must not be swapped.
            for i, collection in enumerate(collections):
                artist_label = collection.get_label()
                if artist_label in expected_by_label:
                    assert artist_label == matched_label_by_index[i], (
                        f"{ax.get_title()} branch label {artist_label!r} is attached "
                        f"to {matched_label_by_index[i]!r} rows"
                    )

            # Union lock: no duplicate rows and no cross-facet leakage.
            observed_union = np.vstack(observed)
            expected_union = np.vstack(list(expected_by_label.values()))
            assert len(np.unique(observed_union, axis=0)) == len(observed_union)
            np.testing.assert_allclose(
                _sort_rows_xy(observed_union),
                _sort_rows_xy(expected_union),
                rtol=0, atol=0,
            )
        plt.close(fig)

    def test_A6_scatter_weights_vector_facet_refuses_until_semantics_exist(self):
        """v0.2 + main-review disposition: bounded Stage A chooses loud refusal.

        Scatter has no statistical weight owner.  Multi-branch weights must not
        collapse to one scalar rendering with only a warning.
        """
        df = _stage_a_df()
        with pytest.raises((ValueError, NotImplementedError)) as excinfo:
            DFDraw(df).scatter(
                "y:x",
                facet_by="f",
                weights_vector=["w0", "w1"],
                weights_labels=["w0", "w1"],
                vector_compose="outer",
            )
        _assert_loud_refusal(excinfo.value)

    def test_A7_scatter_errorbar_plus_expression_color_is_exact_or_refuses(self):
        """RED-R4: support requires exact c*2 mapping for the rendered rows.

        Loud refusal remains valid while the single-panel errorbar+expression
        color owner is unsupported.  Merely restoring yerr is insufficient.
        """
        df = _stage_a_df()
        try:
            fig, _, _ = DFDraw(df).scatter(
                "y:x", facet_by="f", yerr="ey", color="c * 2", colorbar=True
            )
        except (ValueError, NotImplementedError) as exc:
            _assert_loud_refusal(exc)
            return

        axes = _facet_axes(fig)
        assert len(axes) == 2
        assert len(fig.axes) > len(axes), "colorbar=True should add colorbar/helper axes"
        for ax in axes:
            facet_value = _facet_value(ax)
            facet_df = df[df["f"] == facet_value]
            assert _errorbars(ax), f"{ax.get_title()} dropped requested yerr"

            mapped = [c for c in _path_collections(ax) if c.get_array() is not None]
            assert mapped, f"{ax.get_title()} dropped expression color mapping"

            observed_rows = []
            observed_values = []
            for collection in mapped:
                offsets = _offset_array(collection)
                values = np.asarray(collection.get_array(), dtype=float)
                assert len(offsets) == len(values), (
                    f"{ax.get_title()} color mapping length {len(values)} does not "
                    f"match point count {len(offsets)}"
                )
                observed_rows.append(offsets)
                observed_values.append(values)

            observed_rows = np.vstack(observed_rows)
            observed_values = np.concatenate(observed_values)
            order = np.argsort(observed_rows[:, 0], kind="stable")
            observed_rows = observed_rows[order]
            observed_values = observed_values[order]

            expected = facet_df.sort_values("x")
            np.testing.assert_allclose(
                observed_rows,
                expected[["x", "y"]].to_numpy(),
                rtol=0, atol=0,
                err_msg=f"{ax.get_title()} color-mapped points have wrong row membership",
            )
            np.testing.assert_allclose(
                observed_values,
                (expected["c"] * 2.0).to_numpy(),
                rtol=1e-12, atol=1e-12,
                err_msg=f"{ax.get_title()} mapped array is not the requested c*2 expression",
            )
        plt.close(fig)

    def test_A7s_single_panel_errorbar_plus_expression_color_is_exact_or_refuses(self):
        """Single-panel twin of A7: support must preserve exact color semantics.

        The color-loss defect originates in the canonical single-panel scatter
        owner, so a facet-only repair/refusal cannot close Stage A safely.
        """
        df = _stage_a_df()
        try:
            fig, _, _ = DFDraw(df).scatter(
                "y:x", yerr="ey", color="c * 2", colorbar=True
            )
        except (ValueError, NotImplementedError) as exc:
            text = str(exc).lower()
            assert "color" in text and "err" in text, (
                f"refusal should name color + error bars: {exc}"
            )
            return

        ax = fig.axes[0]
        assert len(fig.axes) == 2, "colorbar=True should add one colorbar axis"
        assert _errorbars(ax), "dropped requested yerr"
        mapped = [c for c in _path_collections(ax) if c.get_array() is not None]
        assert len(mapped) == 1, "expression color mapping dropped"
        rows = _offset_array(mapped[0])
        values = np.asarray(mapped[0].get_array(), dtype=float)
        order = np.argsort(rows[:, 0], kind="stable")
        expected = df.sort_values("x")
        np.testing.assert_allclose(
            rows[order], expected[["x", "y"]].to_numpy(), rtol=0, atol=0
        )
        np.testing.assert_allclose(
            values[order], (expected["c"] * 2.0).to_numpy(),
            rtol=1e-12, atol=1e-12,
        )
        plt.close(fig)

    def test_A8_internal_suppress_color_cycle_never_warns_as_user_kwarg(self):
        df = _stage_a_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig, _, _ = DFDraw(df).profile("[y,y+1]:x", bins=6, group_by="g")
        messages = [str(w.message) for w in caught]
        assert not any("_suppress_color_cycle" in m for m in messages), messages
        plt.close(fig)


# ---------------------------------------------------------------------------
# RED-R2: complete disposition declarations for hand-written boundaries.
#
# The relevant typed->facet surface is every explicit public method parameter
# except self/expr/**kwargs.  Therefore a newly added named parameter fails
# completeness until the coder declares exactly one disposition below.
#
# FORWARDED: must appear at _dispatch_faceted_render.
# CONSUMED: intentionally owned outside the per-cell facet boundary.
# REFUSED: composition is intentionally unsupported and must fail loudly.
# ---------------------------------------------------------------------------

_FACET_DISPOSITIONS = {
    "hist": {
        "FORWARDED": {
            "selection", "bins", "range", "norm", "stats", "title", "xlabel", "ylabel",
            "group_by", "ncols", "sharex", "sharey", "top_k", "auto_title",
            "stat_fields", "nan_policy", "time_format", "cumulative", "weights",
            "facet_by", "facet_by_bins", "facet_by_quantiles",
            "share_x", "share_y", "share_across_figures",
            "selection_vector", "weights_vector", "selection_labels", "weights_labels",
            "selection_categorical", "weights_categorical", "vector_compose", "delta_facet",
            "group_by_bins", "group_by_quantiles", "hist_norm", "min_entries",
            "color", "marker", "hist_errors", "linestyle_cycle",
            "fit", "fit_textbox_kwargs", "summary_fit",
        },
        "CONSUMED": {
            "sample",       # selection/sample owner filters df before facet dispatch
            "save",         # post-render save
            "facet",        # normalized to facet_by='group_by' legacy form
            "normalize_layout",  # meaningful only with normalize (refused under facet)
            "legend", "show_legend",          # post-render figure legend owner
        },
        "REFUSED": {
            "normalize",    # PHASE_13_84: hist normalize= + facet_by is unsupported
        },
        "KNOWN_SILENT_GAP": {"ax", "same"},
    },
    "scatter": {
        "FORWARDED": {
            "selection", "color", "size", "marker", "stats", "title", "auto_title",
            "xlabel", "ylabel", "group_by", "ncols", "sharex", "sharey", "top_k",
            "cmap", "colorbar", "clabel", "jitter", "nan_policy",
            "facet_by", "facet_by_bins", "facet_by_quantiles",
            "share_x", "share_y", "share_across_figures",
            "xerr", "yerr", "time_format",
            "fit", "fit_textbox_kwargs", "summary_fit",
        },
        "CONSUMED": {
            "sample",       # df prefilter
            "save",         # post-render save
            "facet",        # legacy facet normalization
            # subordinate vector metadata is owned by the multi-branch safety
            # gate below and has no independent scalar per-cell meaning.
            "selection_labels", "weights_labels",
            "selection_categorical", "weights_categorical",
            "vector_compose", "delta_facet",
            "legend", "show_legend",
        },
        "REFUSED": {
            # Stage-A bounded disposition: scatter has no safe faceted
            # multi-branch owner yet.  Either can be revisited in a later phase.
            "selection_vector", "weights_vector",
        },
        "KNOWN_SILENT_GAP": {"ax", "same"},
    },
    "profile": {
        "FORWARDED": {
            "selection", "bins", "time_format", "range", "error", "stats", "title",
            "xlabel", "ylabel", "group_by", "ncols", "sharex", "sharey", "top_k",
            "return_data", "min_entries", "group_by_bins", "group_by_quantiles",
            "group_by_scope",  # PHASE_13_95: typed profile -> facet policy owner
            "sort_groups", "weights", "auto_title", "stat_fields", "quantiles",
            "central", "quantile_mode", "quantile_style", "nan_policy",
            "facet_by", "facet_by_bins", "facet_by_quantiles", "facet_label_format",
            "share_x", "share_y", "share_across_figures",
            "selection_vector", "weights_vector", "selection_labels", "weights_labels",
            "selection_categorical", "weights_categorical", "vector_compose", "delta_facet",
            "color", "marker", "markersize", "linestyle_cycle",
            "fit", "fit_textbox_kwargs", "summary_fit",
        },
        "CONSUMED": {
            "sample",        # df prefilter
            "bins2",         # profile2d-only alternate owner
            "save",          # post-render save
            "facet",         # normalized to facet_by='group_by'
            "normalize", "normalize_layout",  # alternate normalize orchestrator
            "legend", "show_legend",          # post-render figure legend owner
        },
        "REFUSED": {
            "same",          # explicit public facet_by + same=True guard
            "stat_list",     # Stage B: facet composition reserved for Stage E
        },
        "KNOWN_SILENT_GAP": {"ax"},
    },
    "hist2d": {
        "FORWARDED": {
            "selection", "bins", "range", "norm", "stats", "title", "xlabel", "ylabel",
            "group_by", "ncols", "sharex", "sharey", "top_k",
            "cmap", "colorbar", "clabel", "vmin", "vmax", "auto_title",
            "stat_fields", "nan_policy", "facet_by", "facet_by_bins",
            "facet_by_quantiles", "share_x", "share_y", "share_across_figures",
            "summary_fit", "time_format",
        },
        "CONSUMED": {
            "sample", "save", "facet",
        },
        "REFUSED": {
            "fit",  # explicit Phase-13.51 clean ValueError
        },
        "KNOWN_SILENT_GAP": {"ax", "same"},
    },
}


# ---------------------------------------------------------------------------
# RED-R5: complete draw() -> typed-owner disposition classification.
#
# Applicability is structural: a named draw() parameter is relevant to a typed
# owner when that owner exposes the same explicit parameter.  `profile2d` is a
# public draw(type=...) route whose current executor deliberately delegates to
# profile(), so its structural target is recorded explicitly below.
#
# Every relevant name is exactly one of FORWARDED / CONSUMED / REFUSED.
# `type` and `figsize` are draw-owned routing/materialization controls.
# ---------------------------------------------------------------------------

_DRAW_OWNER_TARGET = {
    "hist": "hist",
    "scatter": "scatter",
    "profile": "profile",
    "hist2d": "hist2d",
    "profile2d": "profile",
}

_DRAW_GLOBAL_CONSUMED = {"type", "figsize"}
_DRAW_SPECIAL_RELEVANT = {
    # profile2d is implemented through profile(**kwargs); norm= is not an
    # explicit profile() parameter but is consumed by the 2D-profile heatmap
    # owner and therefore remains part of the draw-door contract.
    "profile2d": {"norm"},
}

_DRAW_TO_TYPED_DISPOSITIONS = {
    "hist": {
        "FORWARDED": {
            "selection", "group_by", "facet",
            "bins", "stats", "norm", "title", "ax", "sample",
            "save", "same", "nan_policy", "facet_by",
            "facet_by_bins", "facet_by_quantiles", "share_x", "share_y",
            "share_across_figures", "selection_vector", "weights_vector",
            "selection_labels", "weights_labels", "selection_categorical",
            "weights_categorical", "vector_compose", "delta_facet",
            "normalize", "normalize_layout", "fit", "fit_textbox_kwargs",
            "summary_fit", "legend", "show_legend",
        },
        "CONSUMED": set(),
        "REFUSED": set(),
        # B-1: public draw().color/marker are mapping slots, while typed
        # hist().color/marker are literal style slots. Preserve the legacy
        # draw-door gap until an explicit API decision resolves the collision.
        "KNOWN_SILENT_GAP": {"color", "marker"},
    },
    "scatter": {
        "FORWARDED": {
            "selection", "color", "size", "marker", "group_by", "facet",
            "stats", "title", "ax", "sample", "save", "same",
            "nan_policy", "facet_by", "facet_by_bins",
            "facet_by_quantiles", "share_x", "share_y",
            "share_across_figures", "selection_vector", "weights_vector",
            "selection_labels", "weights_labels", "selection_categorical",
            "weights_categorical", "vector_compose", "delta_facet",
            "fit", "fit_textbox_kwargs", "summary_fit", "legend", "show_legend",
        },
        "CONSUMED": set(),
        "REFUSED": set(),
    },
    "profile": {
        "FORWARDED": {
            "selection", "group_by", "group_by_scope", "stat_list", "facet", "bins",
            "stats", "title", "ax", "sample", "save", "same",
            "nan_policy", "facet_by", "facet_by_bins",
            "facet_by_quantiles", "facet_label_format", "share_x", "share_y",
            "share_across_figures", "selection_vector", "weights_vector",
            "selection_labels", "weights_labels", "selection_categorical",
            "weights_categorical", "vector_compose", "delta_facet",
            "normalize", "normalize_layout", "fit", "fit_textbox_kwargs",
            "summary_fit", "legend", "show_legend",
        },
        "CONSUMED": set(),
        "REFUSED": set(),
        # B-1: same public mapping-slot versus typed literal-style collision
        # as hist(). Do not reclassify as forwarding without an API decision.
        "KNOWN_SILENT_GAP": {"color", "marker"},
    },
    "hist2d": {
        "FORWARDED": {
            "selection", "group_by", "facet", "bins", "stats", "norm",
            "title", "ax", "sample", "save", "same", "nan_policy",
            "facet_by", "facet_by_bins", "facet_by_quantiles", "share_x",
            "share_y", "share_across_figures", "summary_fit",
        },
        "CONSUMED": set(),
        "REFUSED": {"fit"},
    },
    "profile2d": {
        "FORWARDED": {
            "selection", "group_by", "facet", "bins", "stats", "norm",
            "title", "ax", "sample", "save", "same", "nan_policy", "facet_by",
            "facet_by_bins", "facet_by_quantiles", "share_x", "share_y",
            "share_across_figures", "selection_vector", "weights_vector",
            "selection_labels", "weights_labels", "selection_categorical",
            "weights_categorical", "vector_compose", "delta_facet",
            "normalize", "normalize_layout", "fit", "fit_textbox_kwargs",
            "summary_fit", "legend", "show_legend",
        },
        "CONSUMED": {
            # PHASE_13_97 Stage A: draw(type="profile2d") owns the scope guard.
            # Explicit nonlegacy formatting is refused before delegation;
            # omitted/legacy formatting needs no forwarding to profile().
            "facet_label_format",
        },
        # profile2d heatmap semantics do not define line/marker styling.  draw()
        # currently drops these silently; RED-R5 requires a loud disposition.
        # Explicit scope is restricted to grouped 1-D profiles.  The
        # profile2d draw-door refuses non-auto scope before allocating a Figure.
        "REFUSED": {"color", "marker", "group_by_scope", "stat_list"},
    },
}


class TestStageAStructuralBoundaryInventory:
    """Mechanical disposition checks for hand-written forwarding boundaries."""

    @staticmethod
    def _explicit_signature_names(method):
        params = inspect.signature(getattr(DFDraw, method)).parameters
        return {
            name for name, param in params.items()
            if name not in {"self", "expr", "kwargs"}
            and param.kind is not inspect.Parameter.VAR_KEYWORD
        }

    @staticmethod
    def _capture_facet_dispatch(method, expr):
        df = _stage_a_df()
        drawer = DFDraw(df)
        captured = {}

        def spy(**kwargs):
            captured.update(kwargs)
            return object(), object(), {}

        drawer._dispatch_faceted_render = spy
        getattr(drawer, method)(expr, facet_by="f")
        return captured

    @pytest.mark.parametrize("method", ["hist", "scatter", "profile", "hist2d"])
    def test_A9a_typed_to_facet_disposition_is_complete_and_exclusive(self, method):
        """RED-R2: every explicit named parameter has exactly one disposition."""
        groups = _FACET_DISPOSITIONS[method]
        declared = set().union(*groups.values())
        expected = self._explicit_signature_names(method)

        overlaps = set().union(*(
            a & b for a, b in itertools.combinations(groups.values(), 2)
        ))
        assert not overlaps, f"{method} multiply classified parameters: {sorted(overlaps)}"
        assert declared == expected, (
            f"{method} disposition drift: "
            f"unclassified={sorted(expected - declared)}, "
            f"stale={sorted(declared - expected)}"
        )

    @pytest.mark.parametrize("method,expr", [
        ("hist", "x"),
        ("scatter", "y:x"),
        ("profile", "y:x"),
        ("hist2d", "y:x"),
    ])
    def test_A9b_typed_to_facet_forwarded_names_reach_dispatch(self, method, expr):
        """RED-R2: every FORWARDED name is structurally present at the boundary."""
        captured = self._capture_facet_dispatch(method, expr)
        expected = _FACET_DISPOSITIONS[method]["FORWARDED"]
        # Some public names are deliberately renamed at the dispatcher
        # boundary; this is semantic forwarding, not loss.
        aliases = {
            "profile": {"range": "x_range"},
        }.get(method, {})
        missing = {
            name for name in expected
            if aliases.get(name, name) not in captured
        }
        assert not missing, (
            f"{method} facet boundary drops FORWARDED parameters: {sorted(missing)}"
        )

    @pytest.mark.parametrize("method,expr", [
        ("hist", "x"),
        ("scatter", "y:x"),
        ("profile", "y:x"),
        ("hist2d", "y:x"),
    ])
    def test_A9c_consumed_names_do_not_leak_to_facet_dispatch(self, method, expr):
        """RED-R2: intentionally outer-owned names stay outside the per-cell boundary."""
        captured = self._capture_facet_dispatch(method, expr)
        leaked = _FACET_DISPOSITIONS[method]["CONSUMED"] & set(captured)
        assert not leaked, (
            f"{method} CONSUMED parameters unexpectedly leaked to facet dispatcher: "
            f"{sorted(leaked)}"
        )

    @pytest.mark.parametrize("vector_kw", ["selection_vector", "weights_vector"])
    def test_A9d_scatter_refused_multi_branch_vector_semantics_are_loud(self, vector_kw):
        """RED-R2: REFUSED scatter-vector dispositions must fail loudly."""
        df = _stage_a_df()
        values = ["sel_left", "sel_mid"] if vector_kw == "selection_vector" else ["w0", "w1"]
        with pytest.raises((ValueError, NotImplementedError)) as excinfo:
            DFDraw(df).scatter("y:x", facet_by="f", **{vector_kw: values})
        _assert_loud_refusal(excinfo.value)

    def test_A9e_profile_refused_same_plus_facet_is_loud(self):
        df = _stage_a_df()
        with pytest.raises(ValueError, match="facet_by.*same=True|same=True.*facet"):
            DFDraw(df).profile("y:x", facet_by="f", same=True)

    def test_A9f_hist2d_refused_fit_is_loud(self):
        df = _stage_a_df()
        with pytest.raises(ValueError, match="does not support fit"):
            DFDraw(df).hist2d("y:x", facet_by="f", fit="gauss")

    def test_A9f2_hist_refused_normalize_plus_facet_is_loud(self):
        df = _stage_a_df()
        with pytest.raises((ValueError, NotImplementedError), match="normalize"):
            DFDraw(df).hist("x", bins=6, facet_by="f", normalize="delta")

    @pytest.mark.parametrize("method,expr,kw", [
        (m, e, k)
        for m, e in [("hist", "x"), ("scatter", "y:x"),
                     ("profile", "y:x"), ("hist2d", "y:x")]
        for k in ("ax", "same")
        if k in _FACET_DISPOSITIONS[m].get("KNOWN_SILENT_GAP", set())
    ])
    @pytest.mark.xfail(strict=True, reason=(
        "KNOWN_SILENT_GAP: ax= / same=True silently ignored under facet_by; "
        "outside Stage A; disposition pending Architect"))
    def test_A9g_known_silent_gap_is_loud(self, method, expr, kw):
        df = _stage_a_df()
        fig0, ax0 = plt.subplots()
        value = ax0 if kw == "ax" else True
        loud = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                fig, _, _ = getattr(DFDraw(df), method)(
                    expr, facet_by="f", **{kw: value}
                )
                plt.close(fig)
            except (ValueError, NotImplementedError) as exc:
                loud = kw in str(exc)
        loud = loud or any(kw in str(w.message) for w in caught)
        plt.close(fig0)
        assert loud, f"{method}(facet_by=..., {kw}=...) is silently ignored"

    # Existing scalar draw->typed checks are expected GREEN guards.
    @pytest.mark.parametrize("plot_type,expr,required", [
        ("hist", "x", {"selection", "bins", "stats", "norm", "title", "sample", "group_by",
                         "facet", "facet_by", "selection_vector", "weights_vector", "normalize"}),
        ("scatter", "y:x", {"selection", "color", "size", "marker", "stats", "title", "sample",
                              "group_by", "facet", "facet_by", "selection_vector", "weights_vector"}),
        ("profile", "y:x", {"selection", "bins", "stats", "title", "sample", "group_by",
                              "facet_by", "selection_vector", "weights_vector", "normalize"}),
        ("hist2d", "y:x", {"selection", "bins", "stats", "title", "sample", "facet_by"}),
    ])
    def test_A10a_draw_to_typed_plain_expression_green_guard(self, plot_type, expr, required):
        """Existing plain-expression delegation guard; expected GREEN on baseline."""
        df = _stage_a_df()
        drawer = DFDraw(df)
        captured = {}

        def spy(_expr, **kwargs):
            captured.update(kwargs)
            return object(), object(), {}

        setattr(drawer, plot_type, spy)
        drawer.draw(expr, type=plot_type)
        missing = required - set(captured)
        assert not missing, f"draw(type={plot_type!r}) drops named parameters: {sorted(missing)}"

    @staticmethod
    def _draw_explicit_signature_names():
        params = inspect.signature(DFDraw.draw).parameters
        return {
            name for name, param in params.items()
            if name not in {"self", "expr", "kwargs"}
            and param.kind is not inspect.Parameter.VAR_KEYWORD
        }

    @staticmethod
    def _owner_relevant_draw_names(owner):
        target = _DRAW_OWNER_TARGET[owner]
        draw_names = TestStageAStructuralBoundaryInventory._draw_explicit_signature_names()
        target_params = inspect.signature(getattr(DFDraw, target)).parameters
        target_names = {
            name for name, param in target_params.items()
            if name not in {"self", "expr", "kwargs"}
            and param.kind is not inspect.Parameter.VAR_KEYWORD
        }
        return (draw_names & target_names) | _DRAW_SPECIAL_RELEVANT.get(owner, set())

    @pytest.mark.parametrize("owner", ["hist", "scatter", "profile", "hist2d", "profile2d"])
    def test_A10d_draw_to_typed_disposition_is_complete_and_exclusive(self, owner):
        """RED-R5: every relevant named draw() parameter is classified once."""
        groups = _DRAW_TO_TYPED_DISPOSITIONS[owner]
        declared = set().union(*groups.values())
        expected = self._owner_relevant_draw_names(owner)
        overlaps = set().union(*(
            a & b for a, b in itertools.combinations(groups.values(), 2)
        ))
        assert not overlaps, f"{owner} multiply classified draw parameters: {sorted(overlaps)}"
        assert declared == expected, (
            f"{owner} draw->typed disposition drift: "
            f"unclassified={sorted(expected - declared)}, "
            f"stale={sorted(declared - expected)}"
        )

    def test_A10e_draw_signature_has_no_unclassified_global_parameter(self):
        """RED-R5: a new explicit draw() parameter cannot escape all owners."""
        draw_names = self._draw_explicit_signature_names()
        owner_declared = set().union(*(
            set().union(*groups.values())
            for groups in _DRAW_TO_TYPED_DISPOSITIONS.values()
        ))
        classified = owner_declared | _DRAW_GLOBAL_CONSUMED
        assert classified == draw_names, (
            f"draw() global disposition drift: "
            f"unclassified={sorted(draw_names - classified)}, "
            f"stale={sorted(classified - draw_names)}"
        )

    @pytest.mark.parametrize("owner,expr", [
        ("hist", "x"),
        ("scatter", "y:x"),
        ("profile", "y:x"),
        ("hist2d", "y:x"),
        ("profile2d", "z:y:x"),
    ])
    def test_A10f_draw_to_typed_forwarded_names_reach_owner(self, owner, expr):
        """RED-R5: every FORWARDED name reaches the actual typed-owner spy."""
        df = _stage_a_df().assign(z=lambda x: x["y"] + 1.0)
        drawer = DFDraw(df)
        target = _DRAW_OWNER_TARGET[owner]
        captured = {}

        def spy(_expr, **kwargs):
            captured.update(kwargs)
            return object(), object(), {}

        setattr(drawer, target, spy)
        probe_kwargs = {"norm": "log"} if owner == "profile2d" else {}
        drawer.draw(expr, type=owner, **probe_kwargs)
        missing = _DRAW_TO_TYPED_DISPOSITIONS[owner]["FORWARDED"] - set(captured)
        assert not missing, (
            f"draw(type={owner!r}) drops FORWARDED parameters at {target} owner: "
            f"{sorted(missing)}"
        )

    @pytest.mark.parametrize("owner,param,value", [
        ("hist2d", "fit", "gauss"),
        ("profile2d", "color", "magenta"),
        ("profile2d", "marker", "s"),
        ("profile2d", "group_by_scope", "global"),
    ])
    def test_A10g_draw_to_typed_refused_names_fail_loudly(self, owner, param, value):
        """RED-R5: REFUSED draw-door semantics cannot be silently discarded."""
        df = _stage_a_df().assign(z=lambda x: x["y"] + 1.0)
        expr = "y:x" if owner == "hist2d" else "z:y:x"
        with pytest.raises((ValueError, NotImplementedError)) as excinfo:
            DFDraw(df).draw(expr, type=owner, **{param: value})
        assert param in str(excinfo.value), (
            f"draw(type={owner!r}) refusal should name {param!r}: {excinfo.value}"
        )

    def test_A10b_draw_profile_normalize_vector_special_delegation_preserves_request(self):
        """RED-R3: execute the special vector+normalize direct delegation.

        This test specifically catches the reviewer mutation that removes
        stats=stats (and analogous named parameters) from the direct
        draw()->profile() vector delegation.
        """
        df = _stage_a_df()
        drawer = DFDraw(df)
        calls = []

        def spy(expr, **kwargs):
            calls.append((expr, kwargs))
            return object(), object(), {}

        drawer.profile = spy
        drawer.draw(
            "[y,y+1]:x",
            type="profile",
            normalize="delta",
            facet_by="f",
            stats=["mean"],
            selection="x >= 0",
            bins=7,
            selection_vector=["sel_left", "sel_mid"],
            weights_vector=["w0", "w1"],
            vector_compose="outer",
        )

        assert len(calls) == 1, f"special normalize delegation should call profile once, got {len(calls)}"
        expr, kwargs = calls[0]
        assert expr == "[y,y+1]:x", f"vector expression was prematurely scalarized: {expr!r}"
        assert kwargs["stats"] == ["mean"], "stats= was dropped by special vector delegation"
        assert kwargs["facet_by"] == "f"
        assert kwargs["normalize"] == "delta"
        assert kwargs["selection"] == "x >= 0"
        assert kwargs["bins"] == 7
        assert kwargs["selection_vector"] == ["sel_left", "sel_mid"]
        assert kwargs["weights_vector"] == ["w0", "w1"]
        assert kwargs["vector_compose"] == "outer"

    def test_A10c_draw_profile_facet_vector_preserves_intact_expression_for_typed_owner(self):
        """Stage B-a lock: facet_by='vector' must reach profile intact.

        This deliberately reaches the production-special bracket path rather
        than the four scalar A10 guards.  Stage B-a makes this a permanent
        GREEN delegation invariant; ordinary facet composition remains B-b.
        """
        df = _stage_a_df()
        drawer = DFDraw(df)
        calls = []

        def spy(expr, **kwargs):
            calls.append((expr, kwargs))
            return object(), object(), {}

        probe_fig, probe_ax = plt.subplots()

        def spy_with_axes(expr, **kwargs):
            calls.append((expr, kwargs))
            return probe_fig, probe_ax, {}

        drawer.profile = spy_with_axes
        drawer.draw(
            "[y,y+1]:x",
            type="profile",
            facet_by="vector",
            stats=["mean"],
            group_by="g",
        )

        try:
            assert len(calls) == 1, (
                "draw() must delegate one intact vector request to profile(); "
                f"got {len(calls)} scalar/partial calls"
            )
            expr, kwargs = calls[0]
            assert expr == "[y,y+1]:x", f"vector expression was prematurely scalarized: {expr!r}"
            assert kwargs["facet_by"] == "vector"
            assert kwargs["stats"] == ["mean"]
            assert kwargs["group_by"] == "g"
        finally:
            plt.close(probe_fig)

    @pytest.mark.parametrize("surface", ["typed", "draw"])
    @pytest.mark.parametrize("plot_type,expr", [
        ("hist", "x"),
        ("scatter", "y:x"),
        ("profile", "y:x"),
        ("hist2d", "y:x"),
    ])
    def test_A12_legacy_facet_true_does_not_leak_facet_layout_controls(
        self, surface, plot_type, expr
    ):
        """B-2: legacy group_by + facet=True must render through both doors.

        The legacy route is normalized to facet_by='group_by'.  Its internal
        facet_by_bins/facet_by_quantiles controls must be consumed by the
        dispatcher, never forwarded into low-level matplotlib kwargs.  The
        historical failure was an AttributeError from matplotlib on
        ``facet_by_bins``.
        """
        df = _stage_a_df()
        drawer = DFDraw(df)
        kwargs = {"group_by": "g", "facet": True}
        if plot_type in {"hist", "profile", "hist2d"}:
            kwargs["bins"] = 6

        if surface == "typed":
            fig, axes, stats = getattr(drawer, plot_type)(expr, **kwargs)
        else:
            fig, axes, stats = drawer.draw(expr, type=plot_type, **kwargs)

        try:
            assert stats.get("faceted") is True, (
                f"{surface} {plot_type} did not report faceted legacy rendering: {stats}"
            )
            assert stats.get("n_groups") == 3, (
                f"{surface} {plot_type} expected 3 legacy facet groups, got {stats}"
            )
        finally:
            plt.close(fig)

    def test_A11_internal_vector_control_is_registered_with_typo_guard(self):
        known = DFDraw._c7_known_kwargs()
        assert "_suppress_color_cycle" in known, (
            "Internal _suppress_color_cycle is injected by _draw_vector but is "
            "absent from the C-7 known-kwarg set"
        )
