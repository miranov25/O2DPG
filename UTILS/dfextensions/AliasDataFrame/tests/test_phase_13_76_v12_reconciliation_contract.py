"""PHASE_13_76_ADF B3.2b — tests-first planner/reconciliation contract.

The intended semantics are fixed by Architect Clarification v05 A–F.
This checkpoint changes NO production code.  Confirmed current B3 defects are
strict xfails so their exact executable shapes are banked before v1.2.
"""

import warnings
import importlib

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:  # package-style
    A_mod = importlib.import_module("AliasDataFrame.AliasDataFrame")
    AliasDataFrame = A_mod.AliasDataFrame
except (ImportError, ModuleNotFoundError):
    A_mod = importlib.import_module("AliasDataFrame")
    AliasDataFrame = A_mod.AliasDataFrame

try:  # production package path
    from dfextensions.dfdraw import DFDraw  # noqa: F401
    HAVE_DFDRAW = True
except Exception:  # pragma: no cover - environment probe
    HAVE_DFDRAW = False

needs_dfdraw = pytest.mark.skipif(
    not HAVE_DFDRAW, reason="dfdraw not importable via dfextensions.dfdraw")

B3_P0_1 = (
    "B3-P0-1: plan omits transitive owner-qualified child aliases that "
    "execution legitimately materializes/cleans"
)
B3_P0_2 = (
    "B3-P0-2: owner-level observation lets one valid same-owner request hide "
    "failure of another valid request"
)
B3_P0_3 = (
    "B3-P0-3: metadata-poor bad-only warn discards exact runtime unresolved "
    "evidence and terminal reconciliation false-refuses"
)
def _parent_child_three_columns():
    parent = AliasDataFrame(pd.DataFrame({
        "k": np.array([0, 1, 2, 3], dtype=np.int64),
        "x": np.array([10.0, 20.0, 30.0, 40.0]),
    }))
    child = AliasDataFrame(pd.DataFrame({
        "k": np.array([0, 1, 2, 3], dtype=np.int64),
        "v": np.array([1.0, 2.0, 3.0, 4.0]),
        "w": np.array([5.0, 6.0, 7.0, 8.0]),
        "u": np.array([9.0, 10.0, 11.0, 12.0]),
    }))
    parent.register_subframe("S", child, index_columns=["k"])
    return parent, child


def _fault_leaf(monkeypatch, parent, leaf):
    original = parent._extract_subframe_values_cached

    def _fault(sf_name, sf_col, indices, missing_mask, *args, **kwargs):
        if sf_name == "S" and sf_col == leaf:
            raise RuntimeError(f"injected valid-request failure for S.{leaf}")
        return original(sf_name, sf_col, indices, missing_mask, *args, **kwargs)

    monkeypatch.setattr(parent, "_extract_subframe_values_cached", _fault)


def _public_draw(parent, specs, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return parent.draw_batch(specs, verbose=False, **kwargs)
        finally:
            plt.close("all")


def _require_reconciliation_rejection(call, *, defect):
    """Pass only when the public call rejects through STEP-9 reconciliation.

    Current B3-P0-2 false greens return normally.  Convert only that exact
    wrong semantic outcome into AssertionError so unrelated runtime failures
    cannot be swallowed by the strict-xfail marker.
    """
    try:
        call()
    except RuntimeError as exc:
        if "[draw-plan-reconcile]" not in str(exc):
            raise
        return
    raise AssertionError(f"{defect}: faulted valid request was silently accepted")


def _convert_known_false_refusal(exc):
    """Convert only the audited B3-P0-3 terminal false-refusal to AssertionError."""
    msg = str(exc)
    required = (
        "[draw-plan-reconcile]",
        "subframes: planned identities not measured",
        "joins: planned identities not measured",
    )
    if all(token in msg for token in required):
        raise AssertionError(
            "B3-P0-3: metadata-poor warn/skip request false-refused by terminal reconciliation"
        ) from exc
    raise exc


@needs_dfdraw
class TestV12B3ReconciliationContract:
    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=B3_P0_1)
    @pytest.mark.parametrize("depth", [2, 3, 4])
    def test_r1_child_alias_dependency_closure_is_planned(self, depth):
        parent, child = _parent_child_three_columns()
        previous = "v"
        expected = []
        for level in range(1, depth + 1):
            name = f"a{level}"
            child.add_alias(name, f"{previous}*2+{level}")
            expected.append(f"S::{name}")
            previous = name

        # Intended public behavior: no reconciliation exception under defaults.
        try:
            _public_draw(parent, {"p": {"expr": f"S.a{depth}:x", "type": "scatter"}})
        except RuntimeError as exc:
            msg = str(exc)
            if ("[draw-plan-reconcile]" in msg
                    and "cleanup: unexpected aliases were dropped" in msg):
                raise AssertionError(
                    f"B3-P0-1: transitive child-alias cleanup was omitted from the plan: {msg}"
                ) from exc
            raise

        plan = parent._last_draw_plan
        state = parent._last_draw_prep_state
        assert set(expected) <= set(plan.aliases), (expected, plan.aliases)
        assert set(expected) <= set(plan.cleanup), (expected, plan.cleanup)
        assert set(expected) <= set(state.aliases_materialized), state.aliases_materialized
        assert set(expected) <= set(state.aliases_dropped), state.aliases_dropped
        assert parent._reconcile_draw_plan_state(plan, state, raise_on_error=False) == ()

    def test_r1_negative_control_one_level_child_alias_remains_green(self):
        parent, child = _parent_child_three_columns()
        child.add_alias("a1", "v*2+1")
        _public_draw(parent, {"p": {"expr": "S.a1:x", "type": "scatter"}})
        plan = parent._last_draw_plan
        state = parent._last_draw_prep_state
        assert "S::a1" in plan.aliases
        assert "S::a1" in state.aliases_dropped
        assert parent._reconcile_draw_plan_state(plan, state, raise_on_error=False) == ()

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=B3_P0_2)
    def test_r2_two_valid_same_owner_requests_cannot_hide_one_runtime_failure(self, monkeypatch):
        parent, _ = _parent_child_three_columns()
        _fault_leaf(monkeypatch, parent, "w")

        # S.v succeeds and supplies owner-level observed S. S.w is also valid
        # but is faulted. Intended request-level accounting must refuse.
        _require_reconciliation_rejection(
            lambda: _public_draw(
                parent,
                {
                    "good": {"expr": "S.v:x", "type": "scatter"},
                    "faulted_valid": {"expr": "S.w:x", "type": "scatter"},
                },
                on_subframe_error="warn",
                on_error="skip",
            ),
            defect="B3-P0-2",
        )

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=B3_P0_3)
    def test_r3_metadata_free_bad_only_warn_uses_runtime_unresolved_evidence(self, tmp_path):
        uproot = pytest.importorskip("uproot")
        if getattr(uproot, "__version__", "") == "stub":
            pytest.skip("real uproot unavailable in local audit environment")
        path = tmp_path / "v12_nometa_bad_only.root"
        with uproot.recreate(str(path)) as f:
            f.mktree("tree", {"k": "int64", "x": "float64"})
            f["tree"].extend({
                "k": np.arange(4, dtype=np.int64),
                "x": np.arange(4, dtype=np.float64) + 10.0,
            })
            f.mktree("Ch", {"k": "int64", "v": "float64"})
            f["Ch"].extend({
                "k": np.arange(4, dtype=np.int64),
                "v": np.arange(4, dtype=np.float64) + 1.0,
            })

        parent = AliasDataFrame.read_tree_lazy(str(path), "tree")
        parent.register_subframe_lazy(
            "S", str(path), tree_name="Ch", index_columns=["k"])

        try:
            result = _public_draw(
                parent,
                {"bad": {"expr": "S.nosuch:x", "type": "scatter"}},
                on_subframe_error="warn",
                on_error="skip",
            )
        except RuntimeError as exc:
            _convert_known_false_refusal(exc)
        assert result is not None
        assert parent._last_draw_prep_state.plan_reconciliation_errors == ()

    def test_r4_mixed_valid_and_bad_warn_normal_execution_passes(self):
        parent, _ = _parent_child_three_columns()
        result = _public_draw(
            parent,
            {
                "good": {"expr": "S.v:x", "type": "scatter"},
                "bad": {"expr": "S.nosuch:x", "type": "scatter"},
            },
            on_subframe_error="warn",
            on_error="skip",
        )
        assert result is not None
        assert parent._last_draw_prep_state.plan_reconciliation_errors == ()

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=B3_P0_2)
    def test_r5_cross_spec_tolerance_cannot_discharge_faulted_valid_spec(self, monkeypatch):
        parent, _ = _parent_child_three_columns()
        _fault_leaf(monkeypatch, parent, "w")

        # spec0 is genuinely unresolved/tolerated, spec1 succeeds and supplies
        # observed owner S, spec2 is a different VALID request and is faulted.
        _require_reconciliation_rejection(
            lambda: _public_draw(
                parent,
                {
                    "bad_tolerated": {"expr": "S.nosuch:x", "type": "scatter"},
                    "good": {"expr": "S.v:x", "type": "scatter"},
                    "faulted_valid": {"expr": "S.w:x", "type": "scatter"},
                },
                on_subframe_error="warn",
                on_error="skip",
            ),
            defect="B3-P0-2",
        )

    @pytest.mark.xfail(strict=True, raises=AssertionError, reason=B3_P0_2)
    def test_r6_cross_slot_success_cannot_hide_faulted_valid_selection(self, monkeypatch):
        parent, _ = _parent_child_three_columns()
        _fault_leaf(monkeypatch, parent, "w")

        # expr request S.v succeeds. selection request S.w is independently
        # valid but faulted. Owner-level S observation must not make it green.
        _require_reconciliation_rejection(
            lambda: _public_draw(
                parent,
                {
                    "p": {
                        "expr": "S.v:x",
                        "selection": "S.w > 0",
                        "type": "scatter",
                    }
                },
                on_subframe_error="warn",
                on_error="skip",
            ),
            defect="B3-P0-2",
        )

    def test_r7_exact_owned_reconciliation_still_rejects_missing_and_extra_owner_effects(self):
        parent, _ = _parent_child_three_columns()
        _public_draw(parent, {"p": {"expr": "S.v:x", "type": "scatter"}})
        plan = parent._last_draw_plan
        state = parent._last_draw_prep_state
        assert parent._reconcile_draw_plan_state(plan, state, raise_on_error=False) == ()

        state.subframes_observed = ()
        state.joins_observed = ()
        missing = parent._reconcile_draw_plan_state(plan, state, raise_on_error=False)
        assert any("subframes: planned identities not measured" in e for e in missing)
        assert any("joins: planned identities not measured" in e for e in missing)

        state.subframes_observed = ("S", "EXTRA")
        state.joins_observed = ("S", "EXTRA")
        extra = parent._reconcile_draw_plan_state(plan, state, raise_on_error=False)
        assert any("subframes: unexpected measured identities" in e for e in extra)
        assert any("joins: unexpected measured identities" in e for e in extra)

    def test_r8_unreferenced_unresolved_alias_does_not_enter_unrelated_plan(self):
        parent = AliasDataFrame(pd.DataFrame({
            "x": np.arange(8, dtype=np.float64),
            "y": np.arange(8, dtype=np.float64) ** 2,
        }))
        parent.add_alias("q", "S.nosuch*2")
        assert "q" in set(parent.validate_aliases())

        _public_draw(parent, {"p": {"expr": "y:x", "type": "scatter"}})
        plan = parent._last_draw_plan
        assert plan.aliases == ()
        assert plan.subframes == ()
        assert plan.joins == ()
        assert "q" in set(parent.validate_aliases())

    def test_r9_projection_failure_precedes_dfdraw_on_error_skip(self):
        parent, _ = _parent_child_three_columns()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError):
                parent.draw_batch(
                    {
                        "good": {"expr": "x", "type": "hist"},
                        "bad": {"expr": "S.nosuch:x", "type": "scatter"},
                    },
                    on_error="skip",
                    verbose=False,
                )
        plt.close("all")

    # B3-P1-1 META-1 is intentionally NOT banked in this tests-only checkpoint.
    # The v01 review proved that its exact assertion depended on v1.1-only
    # private plan fields absent from banked STEP-9.  Reintroduce B3-P1-1 with
    # the production v1.2 increment, where request-level runtime bookkeeping is
    # part of the reviewed implementation substrate.
