"""
Phase 13.30.DF v1.0 — Column reference validation tests.

§9 load-bearing invariance markers per Coder QRC v1.30 Rule 14.

Test classes:
- TestColumnReferenceValidation_Profile  (4 tests; §9.Profile.1..4)
- TestColumnReferenceValidation_Hist     (3 tests; §9.Hist.1..3)
- TestColumnReferenceValidation_Scatter  (2 tests; §9.Scatter.1..2)
- TestR6ColumnReferenceValidator         (2 tests; §9.R6.1..2)
- TestProductionReproducer               (1 test;  §9.Repro.1)

Total: 12 invariance tests.
"""
import inspect
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from dfdraw import DFDraw
from dfdraw.drawer import DFDraw as _DFDraw  # for tuple access in R6 tests


# =============================================================================
# TestColumnReferenceValidation_Profile
# =============================================================================
class TestColumnReferenceValidation_Profile:
    """Class-2 column reference validation on profile()."""

    def _df(self, n=90):
        return pd.DataFrame({
            "x": np.linspace(0, 1, n),
            "y": np.random.RandomState(42).randn(n),
            "row": np.arange(n),
            "sector": np.arange(n) % 3,
        })

    def test_groupby_missing_column_raises_clear_error(self):
        """§9.Profile.1 — group_by='row%3' must raise ValueError with actionable
        message naming the offending param, the bad value, dfdraw context, and
        available columns. Locks BUG_ADF_GroupBy_Expression_Materialization."""
        df = self._df()
        with pytest.raises(ValueError) as exc_info:
            DFDraw(df).profile("y:x", group_by="row%3", bins=10, min_entries=3)

        msg = str(exc_info.value)
        # §9.Profile.1: load-bearing assertions on the message contents
        assert "group_by='row%3'" in msg, f"Missing param=value in msg: {msg!r}"
        assert "profile" in msg, f"Missing context: {msg!r}"
        assert "Available columns" in msg, f"Missing available-cols hint: {msg!r}"
        assert "add_alias" in msg, f"Missing remediation hint: {msg!r}"

    def test_groupby_existing_column_unchanged(self):
        """§9.Profile.2 — group_by='sector' (real column) still works."""
        df = self._df()
        # §9.Profile.2: must NOT raise; must produce grouped output
        fig, ax, stats = DFDraw(df).profile(
            "y:x", group_by="sector", bins=10, min_entries=3,
        )
        # Three sectors → ≥3 lines on the axes
        assert len(ax.get_lines()) >= 3, (
            f"Expected ≥3 lines for 3 sectors; got {len(ax.get_lines())}"
        )

    def test_groupby_none_unchanged(self):
        """§9.Profile.3 — group_by=None (ungrouped path) unaffected."""
        df = self._df()
        # §9.Profile.3: must NOT raise; must produce single-line output
        fig, ax, stats = DFDraw(df).profile("y:x", bins=10, min_entries=1)
        assert stats["n"] == len(df), f"Expected n={len(df)}, got {stats['n']}"

    def test_weights_expression_still_works(self):
        """§9.Profile.4 — Class-4 weights (expression-or-column) unchanged.
        Regression lock: Phase 13.30 must NOT break the weights expression
        contract by accidentally validating weights as Class-2."""
        df = self._df()
        # §9.Profile.4: weights expression like "(1+x**2)" must NOT raise
        fig, ax, stats = DFDraw(df).profile(
            "y:x", weights="(1+x**2)", bins=10, min_entries=3,
        )
        # The call must succeed (no ValueError from Class-2 validator)
        assert "n" in stats


# =============================================================================
# TestColumnReferenceValidation_Hist
# =============================================================================
class TestColumnReferenceValidation_Hist:
    """Class-2 column reference validation on hist()."""

    def _df(self, n=90):
        return pd.DataFrame({
            "y": np.random.RandomState(42).randn(n),
            "row": np.arange(n),
            "sector": np.arange(n) % 3,
        })

    def test_hist_groupby_missing_raises(self):
        """§9.Hist.1 — hist(group_by='row%3') must raise ValueError."""
        df = self._df()
        with pytest.raises(ValueError) as exc_info:
            DFDraw(df).hist("y", group_by="row%3")
        msg = str(exc_info.value)
        # §9.Hist.1
        assert "group_by='row%3'" in msg, f"Missing param=value: {msg!r}"
        assert "hist" in msg, f"Missing context: {msg!r}"

    def test_hist_groupby_existing_unchanged(self):
        """§9.Hist.2 — hist(group_by='sector') still works (real column)."""
        df = self._df()
        # §9.Hist.2
        fig, ax, stats = DFDraw(df).hist("y", group_by="sector")
        assert stats["n"] == len(df)

    def test_hist2d_no_groupby_param_unchanged(self):
        """§9.Hist.3 — hist2d has no group_by parameter → not affected.
        Regression lock: hist2d still works without group_by-related errors."""
        df = self._df()
        df["x"] = np.linspace(0, 1, len(df))
        # §9.Hist.3 — must not raise; hist2d signature has no group_by
        fig, ax, stats = DFDraw(df).hist2d("y:x", bins=10)
        assert "n" in stats


# =============================================================================
# TestColumnReferenceValidation_Scatter
# =============================================================================
class TestColumnReferenceValidation_Scatter:
    """Class-2 column reference validation on scatter()."""

    def _df(self, n=60):
        return pd.DataFrame({
            "x": np.linspace(0, 1, n),
            "y": np.random.RandomState(42).randn(n),
            "row": np.arange(n),
            "sector": np.arange(n) % 3,
        })

    def test_scatter_groupby_missing_raises(self):
        """§9.Scatter.1 — scatter(group_by='row%3') must raise ValueError."""
        df = self._df()
        with pytest.raises(ValueError) as exc_info:
            DFDraw(df).scatter("y:x", group_by="row%3")
        msg = str(exc_info.value)
        # §9.Scatter.1
        assert "group_by='row%3'" in msg, f"Missing param=value: {msg!r}"
        assert "scatter" in msg, f"Missing context: {msg!r}"

    def test_scatter_groupby_existing_unchanged(self):
        """§9.Scatter.2 — scatter(group_by='sector') still works."""
        df = self._df()
        # §9.Scatter.2
        fig, ax, stats = DFDraw(df).scatter("y:x", group_by="sector")
        assert stats["n"] == len(df)


# =============================================================================
# TestR6ColumnReferenceValidator
# =============================================================================
class TestR6ColumnReferenceValidator:
    """R6 class-load validator coverage for new Class-2 tuples."""

    def test_r6_catches_invented_param_in_tuple(self, monkeypatch):
        """§9.R6.1 — adding a fake entry to _PROFILE_COLUMN_REFERENCES must
        cause _validate_forwarded_names() to raise RuntimeError. Locks the
        drift-protection contract for Class-2 tuples."""
        from dfdraw import drawer as drawer_mod

        # Snapshot real tuple
        real_tuple = _DFDraw._PROFILE_COLUMN_REFERENCES
        # Inject a bogus name
        monkeypatch.setattr(_DFDraw, "_PROFILE_COLUMN_REFERENCES",
                            real_tuple + ("no_such_param_at_all",))
        # §9.R6.1: validator must raise
        with pytest.raises(RuntimeError) as exc_info:
            drawer_mod._validate_forwarded_names()
        assert "no_such_param_at_all" in str(exc_info.value)

    def test_r6_tuples_are_subset_of_forwarded(self):
        """§9.R6.2 — every name in _*_COLUMN_REFERENCES must also be present
        in the same plot type's _*_FORWARDED_NAMES OR be a named parameter
        of the method (so kwargs actually reach the validator)."""
        # §9.R6.2
        for tup, method, name in [
            (_DFDraw._PROFILE_COLUMN_REFERENCES, _DFDraw.profile, "profile"),
            (_DFDraw._HIST_COLUMN_REFERENCES,    _DFDraw.hist,    "hist"),
            (_DFDraw._SCATTER_COLUMN_REFERENCES, _DFDraw.scatter, "scatter"),
            (_DFDraw._DRAW_COLUMN_REFERENCES,    _DFDraw.draw,    "draw"),
        ]:
            sig_params = set(inspect.signature(method).parameters)
            for entry in tup:
                assert entry in sig_params, (
                    f"{name}: {entry!r} in _COLUMN_REFERENCES but not in "
                    f"method signature ({sorted(sig_params)})"
                )


# =============================================================================
# TestProductionReproducer
# =============================================================================
class TestProductionReproducer:
    """Lock the exact In[103] production reproducer pattern."""

    def test_production_reproducer_now_raises_not_silent(self):
        """§9.Repro.1 — the synthesized analogue of Marian's In[103] call:

            adf.draw("dy:dedge", selection=..., type="profile",
                     bins=30, group_by="row%3", group_by_bins=5,
                     min_entries=30, range=(0, 5))

        must raise ValueError (not silently degrade to ungrouped + emit
        downstream matplotlib UserWarning, as it did pre-Phase-13.30)."""
        n = 200
        df = pd.DataFrame({
            "dy_I3T":     np.random.RandomState(42).randn(n),
            "dedgeTPCC":  np.linspace(0, 5, n),
            "detType":    np.ones(n, dtype=int),
            "sec":        np.zeros(n, dtype=int),
            "mP4":        np.linspace(-1, 1, n),
            "mP3":        np.linspace(-1, 1, n),
            "row":        np.arange(n),
        })
        # §9.Repro.1 — same call shape as production In[103]:
        with pytest.raises(ValueError) as exc_info:
            DFDraw(df).profile(
                "dy_I3T:dedgeTPCC",
                selection="(detType==1)&(sec==0)&(abs(mP4)<2)&(abs(mP3)<1.4)",
                bins=30,
                group_by="row%3",          # ← computed expression — was silently dropped
                group_by_bins=5,
                min_entries=10,
                auto_title=True,
                range=(0, 5),              # public API uses `range`; mapped to x_range internally
            )
        # The actionable error message tells the user exactly what to do
        msg = str(exc_info.value)
        assert "group_by='row%3'" in msg
        assert "add_alias" in msg or "materialize" in msg
