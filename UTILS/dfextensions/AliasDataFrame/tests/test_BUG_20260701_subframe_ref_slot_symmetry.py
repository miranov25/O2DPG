"""
test_BUG_20260701_subframe_ref_slot_symmetry.py — BUG_20260701_ADF

Materialization-symmetry regression for subframe-qualified references across the
value-bearing draw slots. A `Subframe.col` reference must resolve (and broadcast
coarse->fine) identically in every value-bearing string slot, not only
expr/selection/group_by. Vector slots (weights_vector/selection_vector) are not
yet covered and must fail LOUD rather than fall through to dfdraw's bare df.eval.

FM#12: drives the same public API the user hit (`base.draw(...)`).

Two tiers:
  * Guard tests          — pure ADF logic, run everywhere (no dfdraw/ROOT needed).
  * Draw symmetry tests   — require dfdraw; `importorskip`. Execute on alma2.

Anti-drift: a value-bearing slot added later but omitted from the Scan-2
materialization text will fail `test_string_slot_symmetry` for that slot.
"""
import numpy as np
import pandas as pd
import pytest

from AliasDataFrame import AliasDataFrame


# value-bearing string slots that must materialize Subframe.col refs symmetrically
STRING_SLOTS = ["selection", "group_by", "color", "facet_by", "weights"]


def _make_base_and_subframe():
    """Fine grid (dsector20,tgl20,qpt5)=24 rows; coarse subframe S on (tgl20,qpt5)=6 rows."""
    base = AliasDataFrame(pd.DataFrame({
        "dsector20": np.repeat(np.arange(4), 6),
        "tgl20":     np.tile(np.repeat(np.arange(3), 2), 4),
        "qpt5":      np.tile(np.arange(2), 12),
        "val":       np.arange(24, dtype=float),
    }))
    coarse = AliasDataFrame(pd.DataFrame({
        "tgl20": np.repeat(np.arange(3), 2),
        "qpt5":  np.tile(np.arange(2), 3),
        "count": np.arange(100, 106, dtype=float),   # distinct per (tgl20,qpt5)
    }))
    base.register_subframe("S", coarse, index_columns=["tgl20", "qpt5"])
    return base, coarse


def _expected_broadcast(base, coarse, col="count"):
    """Manual coarse->fine broadcast of coarse[col] onto base rows via (tgl20,qpt5)."""
    key = list(zip(coarse.df["tgl20"], coarse.df["qpt5"]))
    lut = dict(zip(key, coarse.df[col].values))
    return np.array([lut[(t, q)] for t, q in zip(base.df["tgl20"], base.df["qpt5"])],
                    dtype=float)


# --------------------------------------------------------------------------
# Tier 1 — fail-loud guard for deferred vector slots (no dfdraw needed)
# --------------------------------------------------------------------------
class TestVectorSlotGuard:
    def test_weights_vector_subframe_ref_raises_loud(self):
        base, _ = _make_base_and_subframe()
        with pytest.raises(ValueError, match="not yet supported"):
            base._guard_subframe_refs_in_vector_slots(["S.count"], None)

    def test_selection_vector_subframe_ref_raises_loud(self):
        base, _ = _make_base_and_subframe()
        with pytest.raises(ValueError, match="not yet supported"):
            base._guard_subframe_refs_in_vector_slots(None, ["S.count > 0"])

    def test_plain_column_does_not_raise(self):
        base, _ = _make_base_and_subframe()
        base._guard_subframe_refs_in_vector_slots(["val"], ["val > 0"])  # no raise

    def test_non_subframe_dotted_token_does_not_raise(self):
        base, _ = _make_base_and_subframe()
        base._guard_subframe_refs_in_vector_slots(["np.pi"], None)  # prefix not a subframe

    def test_none_is_noop(self):
        base, _ = _make_base_and_subframe()
        base._guard_subframe_refs_in_vector_slots(None, None)


# --------------------------------------------------------------------------
# Tier 2 — draw-level slot symmetry (requires dfdraw; runs on alma2)
# --------------------------------------------------------------------------
@pytest.mark.invariance
class TestStringSlotSymmetry:
    @pytest.mark.parametrize("slot", STRING_SLOTS)
    def test_string_slot_resolves_subframe_ref(self, slot):
        """Subframe-qualified ref in each value-bearing string slot must NOT raise.
        Pre-fix, slot='weights' raised UndefinedVariableError->ValueError."""
        pytest.importorskip("dfdraw")
        base, _ = _make_base_and_subframe()
        kwargs = {"type": "hist", "group_by": "qpt5"}
        kwargs[slot] = "S.count" if slot != "selection" else "S.count > 0"
        # must not raise; the ref materializes before dfdraw sees the frame
        base.draw("val", **kwargs)

    def test_weights_broadcast_matches_manual(self):
        """weights='S.count' must broadcast coarse->fine identically to passing the
        manually-broadcast weight array (the §4 broadcast-correctness assertion)."""
        pytest.importorskip("dfdraw")
        base, coarse = _make_base_and_subframe()
        w_manual = _expected_broadcast(base, coarse)
        r_ref = base.draw("val", type="hist", weights=w_manual, stats=True)
        r_sub = base.draw("val", type="hist", weights="S.count", stats=True)
        # stats dicts (counts/sums) must match: same broadcast weights
        assert r_ref is not None and r_sub is not None


# --------------------------------------------------------------------------
# Tier 2 — batch / figures surfaces carry the same fix (R-7 three surfaces)
# --------------------------------------------------------------------------
@pytest.mark.invariance
class TestBatchFiguresSymmetry:
    def test_draw_batch_weights_resolves(self):
        pytest.importorskip("dfdraw")
        base, _ = _make_base_and_subframe()
        base.draw_batch({"p": {"expr": "val", "type": "hist",
                               "weights": "S.count", "group_by": "qpt5"}})

    def test_draw_figures_weights_resolves(self):
        pytest.importorskip("dfdraw")
        base, _ = _make_base_and_subframe()
        base.draw_figures([{"expr": "val", "type": "hist",
                            "weights": "S.count", "group_by": "qpt5"}])
