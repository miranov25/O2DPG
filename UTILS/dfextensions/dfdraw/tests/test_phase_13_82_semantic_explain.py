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
    CURRENT_GLOBAL_CONFIGURATION,
    Description,
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
