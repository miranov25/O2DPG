"""PHASE_13_76_ADF B3.2b — state/history metamorphic invariants.

AdditionalTests v04 role:
  O-1 final-state vs fresh-instance equivalence, split by independent defect leg,
  O-2 warm/cold/partial-materialization invariance,
  O-3 request/spec permutation invariance,
  O-4 per-public-call evidence isolation in BOTH call orders,
  O-5 unresolved -> later-resolvable retry.

These tests are intentionally small and deterministic.  They add state/history
coverage without changing production behavior.
"""

import warnings
import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from AliasDataFrame.AliasDataFrame import (
        AliasDataFrame,
        SubframeColumnAbsenceError,
    )
except (ImportError, ModuleNotFoundError):
    from AliasDataFrame import AliasDataFrame, SubframeColumnAbsenceError

try:
    from dfextensions.dfdraw import DFDraw  # noqa: F401
    HAVE_DFDRAW = True
except Exception:
    HAVE_DFDRAW = False

needs_dfdraw = pytest.mark.skipif(not HAVE_DFDRAW, reason="dfdraw not importable")

DYN_P0_1 = (
    "DYN-P0-1: virtual upstream alias redefinition leaves materialized "
    "descendants stale"
)
DYN_P0_2 = (
    "DYN-P0-2: re-registering a subframe does not invalidate materialized "
    "aliases sourced from it"
)
B3_P0_2 = (
    "B3-P0-2: request-level attribution is too coarse; spec permutation "
    "must not change or hide a valid-request failure"
)


def _base(n=6):
    x = np.arange(1, n + 1, dtype=np.float64)
    return AliasDataFrame(pd.DataFrame({"x": x})), x


def _define_chain(adf, scale=2.0):
    adf.add_alias("a", f"x*{float(scale)!r}")
    adf.add_alias("b", "a+1")
    adf.add_alias("c", "b*3")


def _oracle(x, scale):
    return (x * float(scale) + 1.0) * 3.0


def _nan_topology(values):
    return np.isnan(np.asarray(values, dtype=np.float64))


def _logical_aliases(adf):
    return dict(adf.aliases)


def _logical_subframes(adf):
    return tuple(sorted(adf.list_subframes()))


@pytest.mark.invariance
class TestV12HistoryInvariance:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P0_1)
    def test_o1_alias_redefinition_final_state_matches_fresh_instance(self):
        """O-1 / DYN-P0-1 leg: history must not change the final observable state."""
        long_lived, x = _base()
        long_lived.add_alias("a", "x*2.0")
        long_lived.add_alias("b", "a+1")

        # Force the audited dangerous topology: sink materialized, upstream virtual.
        long_lived.materialize_aliases(names=["b"])
        assert "a" not in long_lived.df.columns
        assert "b" in long_lived.df.columns
        long_lived.add_alias("a", "x*5.0")

        fresh, _ = _base()
        fresh.add_alias("a", "x*5.0")
        fresh.add_alias("b", "a+1")

        got_history = np.asarray(long_lived.eval("b"))
        got_fresh = np.asarray(fresh.eval("b"))
        expected = x * 5.0 + 1.0

        assert _logical_aliases(long_lived) == _logical_aliases(fresh)
        assert _logical_subframes(long_lived) == _logical_subframes(fresh) == ()
        assert got_history.dtype == got_fresh.dtype == expected.dtype
        np.testing.assert_array_equal(_nan_topology(got_history), _nan_topology(got_fresh))
        np.testing.assert_array_equal(got_history, expected)
        np.testing.assert_array_equal(got_fresh, expected)
        np.testing.assert_array_equal(got_history, got_fresh)

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=DYN_P0_2)
    def test_o1_subframe_reregistration_final_state_matches_fresh_instance(self):
        """O-1 / DYN-P0-2 leg: final subframe state, not history, owns the result."""
        n = 5
        sec = np.arange(n, dtype=np.int64)
        x = np.arange(n, dtype=np.float64)
        coeff0 = np.zeros(n, dtype=np.float64)
        coeff1 = np.arange(1, n + 1, dtype=np.float64)

        long_lived = AliasDataFrame(pd.DataFrame({"sec": sec, "x": x}))
        old_child = AliasDataFrame(pd.DataFrame({"sec": sec, "c0": coeff0}))
        long_lived.register_subframe("Coeff", old_child, index_columns=["sec"])
        long_lived.add_alias("correction", "Coeff.c0")
        long_lived.materialize_aliases(names=["correction"])
        np.testing.assert_array_equal(np.asarray(long_lived.df["correction"]), coeff0)

        new_child = AliasDataFrame(pd.DataFrame({"sec": sec, "c0": coeff1}))
        long_lived.register_subframe("Coeff", new_child, index_columns=["sec"])

        fresh = AliasDataFrame(pd.DataFrame({"sec": sec, "x": x}))
        fresh_child = AliasDataFrame(pd.DataFrame({"sec": sec, "c0": coeff1}))
        fresh.register_subframe("Coeff", fresh_child, index_columns=["sec"])
        fresh.add_alias("correction", "Coeff.c0")

        got_history = np.asarray(long_lived.eval("correction"))
        got_fresh = np.asarray(fresh.eval("correction"))

        assert _logical_aliases(long_lived) == _logical_aliases(fresh)
        assert _logical_subframes(long_lived) == _logical_subframes(fresh) == ("Coeff",)
        assert got_history.dtype == got_fresh.dtype == coeff1.dtype
        np.testing.assert_array_equal(_nan_topology(got_history), _nan_topology(got_fresh))
        np.testing.assert_array_equal(got_history, coeff1)
        np.testing.assert_array_equal(got_fresh, coeff1)
        np.testing.assert_array_equal(got_history, got_fresh)

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param("cold", id="cold"),
            pytest.param("upstream_warm", id="upstream_warm"),
            pytest.param(
                "sink_warm_upstream_virtual",
                marks=pytest.mark.xfail(
                    strict=True, raises=AssertionError, reason=DYN_P0_1
                ),
                id="sink_warm_upstream_virtual-DYN-P0-1",
            ),
            pytest.param("fully_warm", id="fully_warm"),
        ],
    )
    def test_o2_materialization_history_converges_to_same_final_oracle(self, history):
        adf, x = _base()
        adf.add_alias("a", "x*2.0")
        adf.add_alias("b", "a+1")

        if history == "upstream_warm":
            adf.materialize_aliases(names=["a"], cleanTemporary=False)
        elif history == "sink_warm_upstream_virtual":
            adf.materialize_aliases(names=["b"])
            assert "a" not in adf.df.columns
            assert "b" in adf.df.columns
        elif history == "fully_warm":
            adf.materialize_aliases(names=["a", "b"], cleanTemporary=False)

        adf.add_alias("a", "x*100.0")
        got = np.asarray(adf.eval("b"))
        expected = x * 100.0 + 1.0
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)


def _parent_child():
    parent = AliasDataFrame(pd.DataFrame({
        "k": np.arange(4, dtype=np.int64),
        "x": np.array([10., 20., 30., 40.]),
    }))
    child = AliasDataFrame(pd.DataFrame({
        "k": np.arange(4, dtype=np.int64),
        "v": np.array([1., 2., 3., 4.]),
        "w": np.array([5., 6., 7., 8.]),
    }))
    parent.register_subframe("S", child, index_columns=["k"])
    return parent, child


def _draw(parent, specs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return parent.draw_batch(specs, verbose=False, **kwargs)
        finally:
            plt.close("all")


def _fault_w(parent, monkeypatch):
    original = parent._extract_subframe_values_cached

    def fault(sf_name, sf_col, indices, missing_mask, *args, **kwargs):
        if sf_name == "S" and sf_col == "w":
            raise RuntimeError("injected valid-request failure S.w")
        return original(sf_name, sf_col, indices, missing_mask, *args, **kwargs)

    monkeypatch.setattr(parent, "_extract_subframe_values_cached", fault)


def _public_good_observable(result):
    """Stable public draw_batch observable; never compare figure identity."""
    assert isinstance(result, dict)
    assert "good" in result
    entry = result["good"]
    assert isinstance(entry, dict) and "stats" in entry
    stats = entry["stats"]
    return {
        "n": int(stats["n"]),
        "mean_x": float(stats["mean_x"]),
        "mean_y": float(stats["mean_y"]),
    }


def _assert_same_public_observable(got, expected):
    assert got["n"] == expected["n"]
    assert got["mean_x"] == pytest.approx(expected["mean_x"])
    assert got["mean_y"] == pytest.approx(expected["mean_y"])


def _clean_call(parent):
    result = _draw(parent, {"good": {"expr": "S.v:x", "type": "scatter"}})
    plan = parent._last_draw_plan
    state = parent._last_draw_prep_state
    assert state.plan_reconciliation_errors == ()
    assert not any("nosuch" in str(x) for x in plan.logical_requirements)
    return _public_good_observable(result), plan, state


def _tolerated_call(parent):
    result = _draw(
        parent,
        {
            "good": {"expr": "S.v:x", "type": "scatter"},
            "bad": {"expr": "S.nosuch:x", "type": "scatter"},
        },
        on_subframe_error="warn",
        on_error="skip",
    )
    plan = parent._last_draw_plan
    state = parent._last_draw_prep_state
    assert state.plan_reconciliation_errors == ()
    assert any("nosuch" in str(x) for x in plan.logical_requirements)
    return _public_good_observable(result), plan, state


@needs_dfdraw
@pytest.mark.invariance
class TestV12RequestHistoryInvariance:
    def test_o3_spec_permutation_preserves_valid_failure_semantics(self, monkeypatch):
        orders = [
            ("good", "faulted"),
            ("faulted", "good"),
        ]
        for order in orders:
            parent, _ = _parent_child()
            _fault_w(parent, monkeypatch)
            all_specs = {
                "good": {"expr": "S.v:x", "type": "scatter"},
                "faulted": {"expr": "S.w:x", "type": "scatter"},
            }
            specs = {name: all_specs[name] for name in order}
            rejected = False
            try:
                _draw(parent, specs, on_subframe_error="warn", on_error="skip")
            except RuntimeError as exc:
                rejected = "draw-plan-reconcile" in str(exc)
            assert rejected, f"order={order} silently accepted a faulted valid request"

    @pytest.mark.parametrize(
        "order",
        [
            pytest.param("tolerated_then_clean", id="tolerated_then_clean"),
            pytest.param("clean_then_tolerated", id="clean_then_tolerated"),
        ],
    )
    def test_o4_tolerated_request_evidence_is_per_public_call(self, order):
        """O-4 prospective guard: future per-call evidence must never leak."""
        parent, _ = _parent_child()

        if order == "tolerated_then_clean":
            _tolerated_call(parent)
            clean_obs, clean_plan, clean_state = _clean_call(parent)

            fresh, _ = _parent_child()
            fresh_obs, fresh_plan, fresh_state = _clean_call(fresh)
            _assert_same_public_observable(clean_obs, fresh_obs)
            assert clean_state.plan_reconciliation_errors == fresh_state.plan_reconciliation_errors == ()
            assert tuple(clean_plan.subframes) == tuple(fresh_plan.subframes)
            assert tuple(clean_plan.joins) == tuple(fresh_plan.joins)
            assert tuple(clean_plan.logical_requirements) == tuple(fresh_plan.logical_requirements)
            assert tuple(clean_plan.subframe_requests) == tuple(fresh_plan.subframe_requests)
            assert tuple(clean_state.subframe_request_outcomes) == tuple(
                fresh_state.subframe_request_outcomes)
            assert len(clean_state.subframe_request_outcomes) == 1
            assert clean_state.subframe_request_outcomes[0][2] == "success"
        else:
            _clean_call(parent)
            tol_obs, tol_plan, tol_state = _tolerated_call(parent)

            fresh, _ = _parent_child()
            fresh_obs, fresh_plan, fresh_state = _tolerated_call(fresh)
            _assert_same_public_observable(tol_obs, fresh_obs)
            assert tol_state.plan_reconciliation_errors == fresh_state.plan_reconciliation_errors == ()
            assert tuple(tol_plan.subframes) == tuple(fresh_plan.subframes)
            assert tuple(tol_plan.joins) == tuple(fresh_plan.joins)
            assert tuple(tol_plan.logical_requirements) == tuple(fresh_plan.logical_requirements)
            assert tuple(tol_plan.subframe_requests) == tuple(fresh_plan.subframe_requests)
            assert tuple(tol_state.subframe_request_outcomes) == tuple(
                fresh_state.subframe_request_outcomes)
            assert sorted(x[2] for x in tol_state.subframe_request_outcomes) == [
                "success", "tolerated_unresolved"]


@pytest.mark.invariance
class TestV12DeferredRetryInvariance:
    def test_o5_unresolved_alias_becomes_resolvable_without_redefinition(self):
        parent, child = _parent_child()
        parent.add_alias("q", "S.future*2")
        assert "q" in set(parent.validate_aliases())

        with pytest.raises(SubframeColumnAbsenceError):
            parent.eval("q")

        # Extend the current logical namespace after the failed use. q itself
        # is NOT redefined; the retry must resolve against current child state.
        child.add_alias("future", "v+10")
        got = np.asarray(parent.eval("q"))
        expected = (np.asarray(child.df["v"]) + 10.0) * 2.0
        assert got.dtype == expected.dtype
        np.testing.assert_array_equal(got, expected)
        assert "q" not in set(parent.validate_aliases())
