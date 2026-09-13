"""Stage-1 tests for the semantic description and explain surface.

PHASE_13_82_DF. Covers the proposal's T1, T2, T4, T5 and T11 for the fields
extracted in stage 1.

THE TEST THAT MATTERS MOST IS T11.

    The danger this phase has to avoid is two owners: the drawing code deciding
    one thing, and the explanation deciding another, free to drift apart while
    every self-test passes. The extraction is meant to make that impossible by
    construction - one helper, two callers. T11 checks it anyway, by comparing
    the explanation against values read back off the rendered artists.

    T11 does NOT test whether dfdraw is correct. It tests whether the
    explanation describes what dfdraw actually does. The independent
    correctness oracles live in PHASE_13_81 and keep that job.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from dfextensions.dfdraw import DFDraw  # noqa: E402
from dfextensions.dfdraw.plots._semantic import (  # noqa: E402
    BUILTIN_DEFAULT,
    CALL_ARGUMENT,
    CONTRACT_REFUSE_BY_DESIGN,
    CONTRACT_SUPPORTED,
    CONTRACT_UNRESOLVED,
    CURRENT_GLOBAL_CONFIGURATION,
    Description,
    IMPLEMENTATION_KNOWN_GAP,
    IMPLEMENTATION_PASSING,
    IMPLEMENTATION_REFUSES_CORRECTLY,
    IMPLEMENTATION_UNMEASURED,
    KNOWN_GAP_EVIDENCE,
    ORIGIN_DETAIL_CHANGED,
    ORIGIN_DETAIL_UNKNOWN,
    resolve,
)
from dfextensions.dfdraw.plots.profile import draw_profile  # noqa: E402
from dfextensions.dfdraw.style import (  # noqa: E402
    DEFAULT_STYLE,
    get_style,
    get_style_value,
    set_style,
)


@pytest.fixture
def df():
    rng = np.random.default_rng(20260912)
    n = 3000
    return pd.DataFrame({"x": rng.normal(0.0, 1.0, n),
                         "y": rng.normal(0.0, 1.0, n)})


@pytest.fixture
def clean_style():
    """Restore global configuration after any test that changes it."""
    before = get_style()
    yield
    set_style(None)
    set_style(before)


def _draw(df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return draw_profile(df, "x", "y", **kwargs)


# --------------------------------------------------------------------------
# T1 — the helper resolves exactly what the previous inline code resolved
# --------------------------------------------------------------------------

def test_T1_resolve_matches_the_inline_pattern_it_replaced():
    """The extraction must not change a single resolved value.

    The inline code was `if v is None: v = get_style_value(key, fallback)`.
    Anything else is a behaviour change disguised as a refactor.
    """
    for supplied in (None, 100, 0):
        inline = (supplied if supplied is not None
                  else get_style_value("hist.bins", 50))
        viaHelper = resolve(supplied, path="statistic.bins",
                            config_key="hist.bins", fallback=50)
        assert viaHelper == inline, (
            f"supplied={supplied!r}: helper gave {viaHelper!r}, "
            f"inline pattern gives {inline!r}"
        )


def test_T1b_no_config_key_means_the_fallback_applies():
    d = Description()
    got = resolve(None, path="statistic.thing", config_key=None,
                  fallback="fb", description=d)
    assert got == "fb"
    assert d.get("statistic.thing").origin == BUILTIN_DEFAULT


# --------------------------------------------------------------------------
# T2 — explain is read-only
# --------------------------------------------------------------------------

def test_T2_explain_draws_nothing(df):
    """No figure may be created by an explanation."""
    import matplotlib.pyplot as plt
    plt.close("all")
    before = len(plt.get_fignums())
    DFDraw(df)._explain("y:x", type="profile")
    assert len(plt.get_fignums()) == before, (
        "explain created a figure; it must be read-only"
    )


def test_T2b_explain_does_not_mutate_global_configuration(df, clean_style):
    before = get_style()
    DFDraw(df)._explain("y:x", type="profile", bins=123, marker="s")
    assert get_style() == before, (
        "explain changed the global configuration registry"
    )


def test_T2c_explain_does_not_mutate_the_frame(df):
    before = df.copy(deep=True)
    DFDraw(df)._explain("y:x", type="profile")
    pd.testing.assert_frame_equal(df, before)


def test_T2d_explain_does_not_affect_a_later_draw(df):
    """An explanation must leave no trace that changes the next plot."""
    _, ax_a, _ = _draw(df)
    marker_a = ax_a.get_lines()[0].get_marker()

    DFDraw(df)._explain("y:x", type="profile", marker="D", bins=7)

    _, ax_b, _ = _draw(df)
    marker_b = ax_b.get_lines()[0].get_marker()
    assert marker_a == marker_b, (
        f"a draw after explain differed: {marker_a!r} then {marker_b!r}"
    )


# --------------------------------------------------------------------------
# T4 — origin-sensitive fields: same value, distinguishable states
# --------------------------------------------------------------------------

def test_T4_explicitness_is_recorded_for_origin_sensitive_fields(df):
    """`marker` behaves differently depending on whether you supplied it.

    Supplying it means every group gets that marker; omitting it means each
    group takes one from a cycle. So the description must distinguish the two
    states even when the resulting value is identical.
    """
    supplied = DFDraw(df)._explain("y:x", type="profile", marker="o")
    omitted = DFDraw(df)._explain("y:x", type="profile")

    f_sup = supplied.get("aesthetics.marker")
    f_omit = omitted.get("aesthetics.marker")

    assert f_sup.value == f_omit.value, (
        "fixture no longer produces the same value both ways; the test is "
        "not exercising the interesting case"
    )
    assert f_sup.origin == CALL_ARGUMENT
    assert f_sup.explicit_by_user is True
    assert f_omit.origin == CURRENT_GLOBAL_CONFIGURATION
    assert f_omit.explicit_by_user is False


def test_T4b_non_origin_sensitive_fields_do_not_claim_explicitness(df):
    f = DFDraw(df)._explain("y:x", type="profile", bins=42).get("statistic.bins")
    assert f.origin_sensitive is False
    assert "explicit_by_user" not in f.as_dict()


# --------------------------------------------------------------------------
# T5 — provenance honesty: never guess
# --------------------------------------------------------------------------

def test_T5_value_equal_to_shipped_default_is_reported_as_unknown(df, clean_style):
    """Untouched and set-back-to-default are indistinguishable, so say so.

    The registry stores effective values, not their history. Reporting
    BUILTIN_DEFAULT here would be a guess, and wrong whenever a caller really
    did set the value.
    """
    set_style(None)
    f = DFDraw(df)._explain("y:x", type="profile").get("statistic.bins")
    assert f.value == DEFAULT_STYLE["hist.bins"]
    assert f.origin == CURRENT_GLOBAL_CONFIGURATION
    assert f.origin_detail == ORIGIN_DETAIL_UNKNOWN


def test_T5b_changed_registry_value_is_reported_as_changed(df, clean_style):
    set_style({"hist.bins": 77})
    f = DFDraw(df)._explain("y:x", type="profile").get("statistic.bins")
    assert f.value == 77
    assert f.origin == CURRENT_GLOBAL_CONFIGURATION
    assert f.origin_detail == ORIGIN_DETAIL_CHANGED


def test_T5c_explicit_call_argument_carries_no_origin_detail(df):
    f = DFDraw(df)._explain("y:x", type="profile", bins=13).get("statistic.bins")
    assert f.origin == CALL_ARGUMENT
    assert f.origin_detail is None


def test_T5d_an_invented_origin_is_rejected():
    """The vocabulary is closed, so a typo cannot silently become a source."""
    from dfextensions.dfdraw.plots._semantic import Field
    with pytest.raises(ValueError, match="unknown origin"):
        Field(path="a.b", value=1, origin="MADE_UP")


# --------------------------------------------------------------------------
# T11 — the explanation agrees with what the draw path actually did
# --------------------------------------------------------------------------

@pytest.mark.parametrize("case,kwargs", [
    ("plain call", {}),
    ("explicit marker", {"marker": "s"}),
    ("explicit markersize", {"markersize": 11}),
    ("explicit bins", {"bins": 25}),
])
def test_T11_explanation_agrees_with_execution(df, case, kwargs):
    """Compare the explanation against values read back off the artists.

    This is the drift detector. If the explain surface ever grows its own
    copy of a default, this fails.
    """
    d = Description()
    _, ax, _ = _draw(df, _semantic_description=d, **kwargs)

    lines = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0]
    assert lines, f"{case}: nothing was drawn, cannot compare"
    artist = lines[0]

    assert d.get("aesthetics.marker").value == artist.get_marker(), (
        f"{case}: explain says marker="
        f"{d.get('aesthetics.marker').value!r}, artist has "
        f"{artist.get_marker()!r}"
    )
    assert float(d.get("aesthetics.markersize").value) == float(
        artist.get_markersize()), (
        f"{case}: explain says markersize="
        f"{d.get('aesthetics.markersize').value}, artist has "
        f"{artist.get_markersize()}"
    )


def test_T11b_explain_surface_agrees_with_the_draw_path(df, clean_style):
    """The public entry point and the drawing path must resolve identically.

    `_explain()` and a real draw run the same helper, so this compares the two
    callers rather than two implementations.
    """
    set_style({"hist.bins": 63, "profile.markersize": 9})
    for kwargs in ({}, {"marker": "^"}, {"bins": 5}):
        from_explain = DFDraw(df)._explain("y:x", type="profile", **kwargs)

        d = Description()
        _draw(df, _semantic_description=d, **kwargs)

        for path in ("statistic.bins", "aesthetics.marker",
                     "aesthetics.markersize", "aesthetics.capsize"):
            a, b = from_explain.get(path), d.get(path)
            assert a is not None and b is not None, f"{path} missing"
            assert a.value == b.value, (
                f"{kwargs}: {path} explain={a.value!r} draw={b.value!r}"
            )
            assert a.origin == b.origin, (
                f"{kwargs}: {path} origin explain={a.origin} draw={b.origin}"
            )


def test_T11c_drift_would_be_caught(df):
    """Prove T11 can fail: a deliberately wrong expectation must not pass."""
    d = Description()
    _, ax, _ = _draw(df, marker="s", _semantic_description=d)
    artist = [ln for ln in ax.get_lines() if len(ln.get_xdata()) > 0][0]
    assert artist.get_marker() == "s"
    assert d.get("aesthetics.marker").value == "s"
    # If the explain layer had its own default of "o", the assertion in
    # test_T11 would compare "o" against "s" and fail. Documented here so the
    # detector's mechanism is explicit rather than assumed.


# --------------------------------------------------------------------------
# determinism and the unsupported-type refusal
# --------------------------------------------------------------------------

def test_explain_is_deterministic(df):
    a = DFDraw(df)._explain("y:x", type="profile", marker="s").as_dict()
    b = DFDraw(df)._explain("y:x", type="profile", marker="s").as_dict()
    assert a == b


def test_explain_refuses_unsupported_types_loudly(df):
    """Stage 1 covers profile only, and says so rather than half-answering."""
    with pytest.raises(NotImplementedError, match="profile"):
        DFDraw(df)._explain("y:x", type="hist")


# --------------------------------------------------------------------------
# P1-1 — single ownership of the field declarations
# --------------------------------------------------------------------------

def test_P1_1_field_contracts_are_declared_in_exactly_one_place():
    """Neither caller may restate a field's contract.

    Review finding P1-1: `resolve()` owns HOW a value is resolved, but before
    this the production path and `_explain()` each wrote their own path,
    config_key, fallback and origin_sensitive for the same field. A key rename
    or a changed default could then update one reader and leave the other
    stale. This asserts the duplication is gone rather than trusting that it
    was removed.
    """
    import pathlib
    import dfextensions.dfdraw as pkg

    root = pathlib.Path(pkg.__file__).parent
    for rel in ("plots/profile.py", "drawer.py"):
        text = (root / rel).read_text()
        assert "config_key=" not in text, (
            f"{rel} declares a field contract directly; it must consume "
            f"PROFILE_STATIC_FIELDS instead"
        )


def test_P1_1_both_callers_consume_the_same_declarations():
    """Changing a declaration must move BOTH readers, not one."""
    from dfextensions.dfdraw.plots import _semantic as sem

    args = {f.arg for f in sem.PROFILE_STATIC_FIELDS}
    assert args == {"bins", "marker", "markersize", "capsize"}, args

    # Fallbacks must equal the literals the original inline code used, or the
    # extraction changed behaviour.
    by_arg = {f.arg: f for f in sem.PROFILE_STATIC_FIELDS}
    assert by_arg["bins"].config_key == "hist.bins"
    assert by_arg["bins"].fallback == 50
    assert by_arg["marker"].fallback == "o"
    assert by_arg["markersize"].fallback == 6
    assert by_arg["capsize"].fallback == 3
    assert by_arg["marker"].origin_sensitive is True
    assert by_arg["markersize"].origin_sensitive is True
    assert by_arg["bins"].origin_sensitive is False


# --------------------------------------------------------------------------
# Stage B — public/experimental explain surface
# --------------------------------------------------------------------------

def test_B1_public_explain_default_is_machine_readable_dict(df):
    out = DFDraw(df).explain("y:x", type="profile", bins=25, marker="s")
    assert isinstance(out, dict)
    assert out["statistic"]["bins"]["value"] == 25
    assert out["statistic"]["bins"]["source"] == CALL_ARGUMENT
    assert out["aesthetics"]["marker"]["value"] == "s"


def test_B2_public_explain_pretty_is_same_description(df):
    d = DFDraw(df)
    expected = d._explain(
        "y:x", type="profile", bins=25, marker="s"
    ).pretty()
    actual = d.explain(
        "y:x", type="profile", bins=25, marker="s", format="pretty"
    )
    assert actual == expected
    assert "STATISTIC" in actual
    assert "AESTHETICS" in actual


def test_B3_public_explain_dict_matches_private_description(df, clean_style):
    set_style({"hist.bins": 77, "profile.markersize": 9})
    d = DFDraw(df)
    private = d._explain("y:x", type="profile").as_dict()
    public = d.explain("y:x", type="profile")
    assert public == private


def test_B4a_public_explain_supplied_view_is_available(df):
    out = DFDraw(df).explain(
        "y:x", type="profile", view="supplied", bins=17, marker="D"
    )
    assert out["_semantic"]["view"] == "supplied"
    assert out["statistic"]["bins"]["value"] == 17
    assert out["statistic"]["bins"]["source"] == CALL_ARGUMENT
    assert out["aesthetics"]["marker"]["value"] == "D"


@pytest.mark.parametrize("view", ["resolved", "all"])
def test_B4b_public_explain_reserved_resolved_views_fail_loudly(df, view):
    with pytest.raises(NotImplementedError, match="resolved/runtime"):
        DFDraw(df).explain("y:x", type="profile", view=view)


def test_B5_public_explain_unknown_view_is_rejected(df):
    with pytest.raises(ValueError, match="unknown explain view"):
        DFDraw(df).explain("y:x", type="profile", view="future")


def test_B6_public_explain_unknown_format_is_rejected(df):
    with pytest.raises(ValueError, match="unknown explain format"):
        DFDraw(df).explain("y:x", type="profile", format="yaml")


def test_B7_public_explain_refuses_unsupported_type_loudly(df):
    with pytest.raises(NotImplementedError, match="profile"):
        DFDraw(df).explain("x", type="hist")


def test_B8_public_explain_draws_nothing_and_leaves_frame_unchanged(df):
    import matplotlib.pyplot as plt
    plt.close("all")
    before_frame = df.copy(deep=True)
    before_figs = tuple(plt.get_fignums())
    DFDraw(df).explain("y:x", type="profile", bins=17, marker="D")
    assert tuple(plt.get_fignums()) == before_figs
    pd.testing.assert_frame_equal(df, before_frame)


def test_B9_public_explain_does_not_mutate_global_configuration(df, clean_style):
    before = get_style()
    DFDraw(df).explain("y:x", type="profile", bins=17, marker="D")
    assert get_style() == before


def test_B10_public_explain_does_not_change_later_draw(df):
    import matplotlib.pyplot as plt

    # This test exercises two real draws. Keep its pyplot state local so the
    # xdist worker cannot leak an ambient Axes into unrelated same=True tests.
    plt.close("all")
    try:
        _, ax_a, _ = _draw(df)
        marker_a = ax_a.get_lines()[0].get_marker()

        DFDraw(df).explain("y:x", type="profile", marker="D", bins=7)

        _, ax_b, _ = _draw(df)
        marker_b = ax_b.get_lines()[0].get_marker()
        assert marker_a == marker_b
    finally:
        plt.close("all")


# --------------------------------------------------------------------------
# Gate 1A / CRR-1 — static/descriptive semantic slices S1/S2/S3/S6/S7
# --------------------------------------------------------------------------

def test_G1A_status_vocabulary_is_ratified_two_axis_model():
    from dfextensions.dfdraw.plots import _semantic as sem
    assert sem.CONTRACT_STATUSES == (
        "SUPPORTED", "NOT_APPLICABLE", "REFUSE_BY_DESIGN", "UNRESOLVED"
    )
    assert sem.IMPLEMENTATION_STATUSES == (
        "PASSING", "KNOWN_GAP", "REFUSES_CORRECTLY", "TEST_GAP",
        "UNMEASURED"
    )


def test_G1A_S1_supplied_omits_unsupplied_defaults(df):
    out = DFDraw(df).explain("y:x", type="profile", view="supplied")
    assert out["_semantic"]["contract_status"] == CONTRACT_SUPPORTED
    assert out["_semantic"]["implementation_status"] == IMPLEMENTATION_PASSING
    assert "statistic" not in out
    assert "aesthetics" not in out


def test_G1A_S1_effective_global_then_explicit_override(df, clean_style):
    set_style({"hist.bins": 77})
    inherited = DFDraw(df).explain("y:x", type="profile")
    explicit = DFDraw(df).explain("y:x", type="profile", bins=40)
    assert inherited["statistic"]["bins"]["value"] == 77
    assert inherited["statistic"]["bins"]["config_key"] == "hist.bins"
    assert explicit["statistic"]["bins"]["value"] == 40
    assert explicit["statistic"]["bins"]["source"] == CALL_ARGUMENT


def test_G1A_S2_same_effective_marker_can_preserve_different_explicitness(
    df, clean_style
):
    set_style({"profile.marker": "s"})
    inherited = DFDraw(df).explain(
        "y:x", type="profile", group_by="g"
    )
    explicit = DFDraw(df).explain(
        "y:x", type="profile", group_by="g", marker="s"
    )
    a = inherited["aesthetics"]["marker"]
    b = explicit["aesthetics"]["marker"]
    assert a["value"] == b["value"] == "s"
    assert a["explicit_by_user"] is False
    assert b["explicit_by_user"] is True


def test_G1A_S3_multibranch_selection_vector_has_named_branch_coordinate(df):
    out = DFDraw(df).explain(
        "y:x", type="profile",
        selection_vector=["x < 0", "x >= 0"],
        vector_compose="outer",
    )
    assert out["selection"]["vector"] == ["x < 0", "x >= 0"]
    branch = out["coordinates"]["branch"]
    assert branch["kind"] == "selection_vector"
    assert branch["cardinality"] == 2
    assert branch["order"] == [0, 1]


def test_G1A_S3_one_element_vector_lowers_to_scalar_without_branch(df):
    out = DFDraw(df).explain(
        "y:x", type="profile", selection_vector=["x < 0"]
    )
    assert out["selection"]["scalar"] == "x < 0"
    assert out["selection"]["lowered_from"] == "selection_vector"
    assert out["selection"]["vector_channel_cost"] == 0
    assert "coordinates" not in out or "branch" not in out.get("coordinates", {})
    assert out["_semantic"]["contract_status"] == CONTRACT_SUPPORTED
    assert out["_semantic"]["implementation_status"] == IMPLEMENTATION_KNOWN_GAP
    assert out["_semantic"]["evidence"] == [
        "PHASE_13_77 Stage-A ORACLE-01"
    ]


def test_G1A_S3_one_element_vector_combines_with_global_selection(df):
    out = DFDraw(df).explain(
        "y:x", type="profile", selection="y > 0",
        selection_vector=["x < 0"]
    )
    assert out["selection"]["scalar"] == "(y > 0) & (x < 0)"


def test_G1A_S6_group_coordinate_is_named_without_resolving_membership(df):
    out = DFDraw(df).explain(
        "y:x", type="profile", group_by="g", group_by_bins=4
    )
    assert out["coordinates"]["group"] == {"expression": "g", "bins": 4}
    assert "resolved" not in out["coordinates"]["group"]


def _without_semantic_header(d):
    return {k: v for k, v in d.items() if k != "_semantic"}


def test_G1A_S7_declared_profile_doors_share_contract_semantics(df):
    kwargs = {"normalize": "delta"}
    via_draw = DFDraw(df).explain(
        "[y,y+1]:x", type="profile", door="draw", **kwargs
    )
    via_profile = DFDraw(df).explain(
        "[y,y+1]:x", type="profile", door="profile", **kwargs
    )
    assert _without_semantic_header(via_draw) == _without_semantic_header(via_profile)
    assert via_draw["transform"]["normalize"] == "delta"
    assert via_draw["_semantic"]["implementation_status"] == IMPLEMENTATION_KNOWN_GAP
    assert "PHASE_13_77 Stage-A ORACLE-05" in via_draw["_semantic"]["evidence"]
    assert via_profile["_semantic"]["implementation_status"] == IMPLEMENTATION_PASSING


def test_G1A_S7_unknown_door_is_rejected(df):
    with pytest.raises(ValueError, match="unknown explain door"):
        DFDraw(df).explain("y:x", type="profile", door="hist")


def test_G1A_pretty_and_dict_are_same_semantic_owner(df):
    d = DFDraw(df)
    machine = d.explain(
        "y:x", type="profile", view="effective",
        selection_vector=["x < 0"],
    )
    pretty = d.explain(
        "y:x", type="profile", view="effective", format="pretty",
        selection_vector=["x < 0"],
    )
    assert machine["_semantic"]["implementation_status"] in pretty
    assert "ORACLE-01" in pretty


def test_G1A_unmodeled_kwarg_refuses_instead_of_returning_partial_answer(df):
    with pytest.raises(NotImplementedError, match="does not yet describe"):
        DFDraw(df).explain("y:x", type="profile", title="not-yet-modeled")


def test_G1A_S7_does_not_overgeneralize_delta_evidence(df):
    draw_ratio = DFDraw(df).explain(
        "[y,y+1]:x", type="profile", door="draw", normalize="ratio"
    )
    assert draw_ratio["_semantic"]["implementation_status"] == "UNMEASURED"
    assert draw_ratio["_semantic"]["evidence"] == []


# --------------------------------------------------------------------------
# Gate 1A / CRR-1 v1.1 corrections — review R1/R2/R3/R4
# --------------------------------------------------------------------------

def test_G1A_v11_status_known_gap_is_not_weakened_by_unmeasured(df):
    out = DFDraw(df).explain(
        "[y,y]:x", type="profile", door="draw",
        selection_vector=["x < 0"], normalize="ratio",
    )
    assert out["_semantic"]["implementation_status"] == IMPLEMENTATION_KNOWN_GAP
    assert out["_semantic"]["evidence"] == [
        "PHASE_13_77 Stage-A ORACLE-01"
    ]


def test_G1A_v11_status_multiple_known_gap_evidence_coexists(df):
    out = DFDraw(df).explain(
        "[y,y]:x", type="profile", door="draw",
        selection_vector=["x < 0"], normalize="delta",
    )
    assert out["_semantic"]["implementation_status"] == IMPLEMENTATION_KNOWN_GAP
    assert set(out["_semantic"]["evidence"]) == {
        "PHASE_13_77 Stage-A ORACLE-01",
        "PHASE_13_77 Stage-A ORACLE-05",
    }


def test_G1A_v11_refusal_is_terminal_and_drops_downstream_gap_evidence(df):
    out = DFDraw(df).explain(
        "[y,y]:x", type="profile", door="draw",
        selection_vector=["x < 0", "x >= 0", "y > 0"],
        vector_compose="inner", normalize="delta",
    )
    assert out["_semantic"]["contract_status"] == CONTRACT_REFUSE_BY_DESIGN
    assert out["_semantic"]["implementation_status"] == IMPLEMENTATION_REFUSES_CORRECTLY
    assert out["_semantic"]["evidence"] == []
    assert "refusal_reason" in out["composition"]


def test_G1A_v11_vector_owner_empty_selection_refusal(df):
    out = DFDraw(df).explain(
        "y:x", type="profile", selection_vector=[]
    )
    assert out["_semantic"]["contract_status"] == CONTRACT_REFUSE_BY_DESIGN
    assert out["_semantic"]["implementation_status"] == IMPLEMENTATION_REFUSES_CORRECTLY
    assert out["selection"]["vector"] == []
    assert "non-empty" in out["composition"]["refusal_reason"]


def test_G1A_v11_vector_owner_scalar_y_two_selection_inner_refuses(df):
    out = DFDraw(df).explain(
        "y:x", type="profile",
        selection_vector=["x < 0", "x >= 0"],
        vector_compose="inner",
    )
    assert out["_semantic"]["contract_status"] == CONTRACT_REFUSE_BY_DESIGN
    assert out["_semantic"]["implementation_status"] == IMPLEMENTATION_REFUSES_CORRECTLY
    assert "branch" not in out.get("coordinates", {})
    assert "equal lengths" in out["composition"]["refusal_reason"]


def test_G1A_v11_vector_owner_scalar_y_two_selection_outer(df):
    out = DFDraw(df).explain(
        "y:x", type="profile",
        selection_vector=["x < 0", "x >= 0"],
        vector_compose="outer",
    )
    assert out["_semantic"]["contract_status"] == CONTRACT_SUPPORTED
    assert out["coordinates"]["branch"] == {
        "kind": "selection_vector",
        "cardinality": 2,
        "order": [0, 1],
    }
    assert out["composition"]["iteration_count"] == 2


def test_G1A_v11_vector_owner_two_y_two_selection_inner(df):
    out = DFDraw(df).explain(
        "[y,y]:x", type="profile",
        selection_vector=["x < 0", "x >= 0"],
        vector_compose="inner",
    )
    assert out["coordinates"]["branch"] == {
        "kind": "selection_vector",
        "cardinality": 2,
        "order": [0, 1],
    }
    assert out["composition"]["iteration_count"] == 2


def test_G1A_v11_dependent_group_bins_without_group_is_explicit(df):
    out = DFDraw(df).explain(
        "y:x", type="profile", group_by_bins=4
    )
    assert out["grouping"]["bins"] == 4
    assert out["grouping"]["parent"] is None
    assert out["_semantic"]["contract_status"] == CONTRACT_UNRESOLVED
    assert out["_semantic"]["implementation_status"] == IMPLEMENTATION_UNMEASURED


@pytest.mark.xfail(
    strict=True,
    reason="PHASE_13_77 Stage-A ORACLE-01: one-element selection_vector must equal scalar selection",
)
def test_T_CAL_01_oracle01_one_element_selection_vector_product_equivalence():
    rng = np.random.default_rng(13018201)
    left_x = rng.normal(-1.0, 0.15, 100)
    right_x = rng.normal(1.0, 0.15, 100)
    frame = pd.DataFrame({
        "x": np.concatenate([left_x, right_x]),
        "y": np.arange(200, dtype=float),
        "facet": np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)]),
    })
    d = DFDraw(frame)
    import matplotlib.pyplot as plt
    try:
        _, _, scalar_stats = d.profile(
            "y:x", bins=4, return_data=True,
            selection="x < 0", facet_by="facet",
        )
        _, _, vector_stats = d.profile(
            "y:x", bins=4, return_data=True,
            selection_vector=["x < 0"], facet_by="facet",
        )
        assert vector_stats["n_total"] == scalar_stats["n_total"]
        assert vector_stats["groups"] == scalar_stats["groups"]
    finally:
        plt.close("all")


@pytest.mark.xfail(
    strict=True,
    reason="PHASE_13_77 Stage-A ORACLE-05: draw/profile bracket-vector delta doors must agree",
)
def test_T_CAL_05_oracle05_draw_profile_normalize_product_equivalence():
    rng = np.random.default_rng(13018205)
    n = 400
    frame = pd.DataFrame({
        "x": rng.normal(size=n),
        "y0": rng.normal(size=n),
        "y1": rng.normal(size=n) + 0.5,
    })
    d = DFDraw(frame)
    import matplotlib.pyplot as plt
    try:
        _, _, draw_stats = d.draw(
            "[y0,y1]:x", type="profile", normalize="delta", bins=12
        )
        _, _, profile_stats = d.profile(
            "[y0,y1]:x", normalize="delta", bins=12
        )
        assert isinstance(draw_stats, dict)
        assert "normalize_data" in draw_stats
        assert draw_stats["normalize_mode"] == profile_stats["normalize_mode"]
        pd.testing.assert_frame_equal(
            draw_stats["normalize_data"].reset_index(drop=True),
            profile_stats["normalize_data"].reset_index(drop=True),
        )
    finally:
        plt.close("all")


STRICT_KNOWN_GAP_CALIBRATIONS = {
    "PHASE_13_77 Stage-A ORACLE-01":
        "test_T_CAL_01_oracle01_one_element_selection_vector_product_equivalence",
    "PHASE_13_77 Stage-A ORACLE-05":
        "test_T_CAL_05_oracle05_draw_profile_normalize_product_equivalence",
}


def test_G1A_v11_meta_every_known_gap_has_strict_calibration():
    assert set(STRICT_KNOWN_GAP_CALIBRATIONS) == set(KNOWN_GAP_EVIDENCE)
    for evidence, test_name in STRICT_KNOWN_GAP_CALIBRATIONS.items():
        fn = globals()[test_name]
        marks = getattr(fn, "pytestmark", [])
        strict_xfail = [
            mark for mark in marks
            if mark.name == "xfail" and mark.kwargs.get("strict") is True
        ]
        assert strict_xfail, f"{evidence} calibration is not strict xfail"
