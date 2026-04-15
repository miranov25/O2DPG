"""
Phase 13.19.ADF.FIX1 — K1 Diagnostic Tests for Vector draw() Kwarg Propagation

STANDALONE NEW TEST FILE. Marked @pytest.mark.invariance.

PURPOSE: locate where named kwargs are silently dropped between
adf.draw() (and friends) and the inner DFDraw call. Tests are
DIAGNOSTIC: pass/fail outcome is the deliverable. A failing test is
not a regression — it is a localization of the bug.

PHASE: 13.19.ADF.FIX1 (v0.2 diagnostic-first)
ARCHITECT DIRECTION (2026-04-14):
    "It is not clear where is bug. So I suggested to make a test in ADF
    and dfraw for that bug. I suggest to extend tests first. We will
    find out where is the problem."

DESIGN:
    Each test uses the dfdraw_call_capture fixture to monkey-patch
    DFDraw's plot methods (hist/hist2d/scatter/profile/hexbin) and
    capture their *args/**kwargs. After exercising adf.draw() (or
    draw_batch / draw_figures), tests inspect the captured call
    record to confirm which named parameters reached DFDraw.

NO MODIFICATIONS TO AliasDataFrame.py. This is a diagnostic phase.

EXPECTED OUTCOMES:
    K1_1 should pass (signature surface is well-formed).
    K1_2 may fail — keyword-only locals (lazy, keep_materialized,
         entry_begin, entry_end, entry_mask) are stripped from
         **kwargs by the `*,` separator at line 9995 of
         AliasDataFrame.py and never re-injected before the
         plot_func(**kwargs) call at line 10184. Drop expected.
    K1_3, K1_4: depend on whether draw_batch / draw_figures
         re-inject batch/figure-level kwargs into per-call dispatch.
    K1_5: depends on whether dfdraw expands vector exprs or not.
         If it does, captures multiple calls and we check kwargs
         on each. If it doesn't, captures one call.

Findings to be published in
    PHASE_13_19_ADF_FIX1_v0.2_Diagnostic_Findings.md
after running this file.
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AliasDataFrame import AliasDataFrame


# ============================================================================
# Fixture: monkey-patch DFDraw plot methods to capture call args
# ============================================================================

@pytest.fixture
def dfdraw_call_capture(monkeypatch):
    """
    Monkey-patch every public plot method on DFDraw so each call
    appends a record to a list.

    Yields:
        list of dict, each with keys 'method', 'args', 'kwargs'.
        The list is mutated in-place during the test as DFDraw is
        called. Inspect after the draw operation.

    pytest-skips if dfdraw is not importable.
    """
    try:
        from dfextensions.dfdraw import DFDraw
    except ImportError:
        try:
            from dfdraw import DFDraw
        except ImportError:
            pytest.skip("dfdraw not available")

    calls = []

    plot_methods = ['hist', 'hist2d', 'scatter', 'profile', 'hexbin']

    for method_name in plot_methods:
        if not hasattr(DFDraw, method_name):
            continue
        original = getattr(DFDraw, method_name)

        def _make_wrapper(mname, orig):
            def wrapper(self, *args, **kwargs):
                calls.append({
                    'method': mname,
                    'args': args,
                    'kwargs': dict(kwargs),
                })
                # Try to delegate to original. If original raises
                # (e.g., draw backend missing), still keep the
                # capture record — diagnostic value is in the
                # captured kwargs, not in successful rendering.
                try:
                    return orig(self, *args, **kwargs)
                except Exception as e:
                    # Record failure but don't propagate — diagnostic
                    # only needs to know what was passed in.
                    calls[-1]['inner_exception'] = type(e).__name__
                    return None
            return wrapper

        monkeypatch.setattr(DFDraw, method_name, _make_wrapper(method_name, original))

    yield calls
    # monkeypatch reverts automatically on teardown


# ============================================================================
# Fixture: small AliasDataFrame for diagnostic
# ============================================================================

def _build_small_adf():
    """
    Minimal ADF with x, y1, y2, z columns for diagnostic draw tests.
    Returns ADF with 100 rows of well-behaved float data.
    """
    np.random.seed(13191)
    n = 100
    df = pd.DataFrame({
        'x': np.linspace(0.0, 10.0, n),
        'y1': np.random.randn(n) + np.linspace(0.0, 5.0, n),
        'y2': np.random.randn(n) + np.linspace(5.0, 0.0, n),
        'z': np.random.choice(['A', 'B', 'C'], size=n),
    })
    return AliasDataFrame(df)


# ============================================================================
# Diagnostic Tests
# ============================================================================

class TestK1VectorDrawKwargDiagnostic:
    """
    K1_1 .. K1_5 — diagnose where named kwargs are dropped between
    adf.draw() (and family) and the inner DFDraw call.
    """

    @pytest.mark.invariance
    def test_K1_1_draw_accepts_all_documented_kwargs(self):
        """
        K1_1 SIGNATURE BINDING (sanity, expected pass).

        Every parameter listed in the def draw(...) signature must
        be accepted without TypeError. This is a sanity check on
        the public surface — not a substantive test of forwarding.
        """
        adf = _build_small_adf()

        # Should not raise. All keyword-only params from line
        # 9995 of AliasDataFrame.py + a handful of common dfdraw kwargs.
        try:
            adf.draw(
                "y1:x",
                type='scatter',
                lazy=True,
                keep_materialized=False,
                entry_begin=0,
                entry_end=50,
                entry_mask=None,
                bins=12,
                group_by='z',
                group_by_bins=3,
            )
        except TypeError as e:
            pytest.fail(
                f"K1_1: adf.draw() rejected a documented signature "
                f"parameter — TypeError: {e}"
            )
        except Exception:
            # Other exceptions (e.g., backend rendering issues) are
            # not the concern of K1_1; signature binding is.
            pass

    @pytest.mark.invariance
    def test_K1_2_draw_forwards_kwargs_to_dfdraw(self, dfdraw_call_capture):
        """
        K1_2 INNER DISPATCH (the real diagnostic).

        Every named kwarg passed to adf.draw() must reach the inner
        DFDraw plot method with the value the caller passed.

        DROPS to investigate (per source reading at line 9995):
            lazy, keep_materialized, entry_begin, entry_end, entry_mask
            -- these are keyword-only locals stripped from **kwargs.
            -- dfdraw never sees them unless re-injected before the
               plot_func(**kwargs) call at line 10184.

        If those drops are intentional (ADF consumes them locally
        before delegating), this test should be amended to mark them
        ADF-CONSUMED rather than ADF-DROPPED. The diagnostic publishes
        evidence either way.

        Common "user intent" kwargs that MUST reach DFDraw:
            bins, group_by, group_by_bins, selection, range,
            (and any other dfdraw-recognized rendering parameter)
        """
        adf = _build_small_adf()

        # User-intent kwargs that should clearly reach DFDraw
        user_intent_kwargs = dict(
            bins=17,
            group_by='z',
            group_by_bins=3,
            selection='x > 1.0',
        )
        # ADF-internal kwargs that may be consumed locally (keyword-only)
        adf_local_kwargs = dict(
            lazy=False,
            keep_materialized=False,
        )

        all_kwargs = {**user_intent_kwargs, **adf_local_kwargs}
        adf.draw("y1:x", type='scatter', **all_kwargs)

        assert len(dfdraw_call_capture) >= 1, (
            "K1_2: expected at least 1 DFDraw call; got "
            f"{len(dfdraw_call_capture)}. "
            f"Capture content: {dfdraw_call_capture!r}"
        )

        captured = dfdraw_call_capture[0]['kwargs']
        method_called = dfdraw_call_capture[0]['method']

        # Build a per-kwarg report. Each entry is one of:
        #   ARRIVED: forwarded with same value
        #   ARRIVED-MUTATED: forwarded but value differs
        #   ABSENT: not in captured kwargs
        report_lines = [
            f"K1_2 diagnostic — captured DFDraw.{method_called}() call:"
        ]
        drops = []
        for key, expected in user_intent_kwargs.items():
            if key not in captured:
                drops.append(key)
                report_lines.append(
                    f"  ABSENT (USER-INTENT DROP): {key}={expected!r}"
                )
            elif captured[key] != expected:
                report_lines.append(
                    f"  ARRIVED-MUTATED: {key} sent={expected!r} "
                    f"received={captured[key]!r}"
                )
            else:
                report_lines.append(f"  ARRIVED: {key}={expected!r}")

        for key, expected in adf_local_kwargs.items():
            if key not in captured:
                report_lines.append(
                    f"  ABSENT (ADF-LOCAL, may be intentional): "
                    f"{key}={expected!r}"
                )
            else:
                report_lines.append(
                    f"  ARRIVED (UNEXPECTED — ADF-local kwarg leaked "
                    f"to DFDraw): {key}={captured[key]!r}"
                )

        # Print the full diagnostic report regardless of pass/fail.
        # pytest -s shows it; otherwise it appears on failure.
        print("\n" + "\n".join(report_lines))

        # USER-INTENT drops are the bug class. Fail on those.
        assert not drops, (
            "K1_2 BUG LOCALIZATION: ADF-side dropped these user-intent "
            f"kwargs before reaching DFDraw: {drops}\n"
            + "\n".join(report_lines)
        )

    @pytest.mark.invariance
    def test_K1_3_draw_batch_forwards_batch_kwargs(self, dfdraw_call_capture):
        """
        K1_3 BATCH DISPATCH.

        adf.draw_batch(specs, group_by='z', group_by_bins=4) — does
        each per-spec inner draw call receive group_by and
        group_by_bins?

        draw_batch signature accepts **kwargs; the question is whether
        it threads them into each spec's inner call.
        """
        adf = _build_small_adf()

        specs = {
            'plot_y1': {'expr': 'y1:x', 'type': 'scatter'},
            'plot_y2': {'expr': 'y2:x', 'type': 'scatter'},
        }
        try:
            adf.draw_batch(specs, group_by='z', group_by_bins=4, bins=10)
        except Exception as e:
            print(f"\nK1_3 note: draw_batch raised {type(e).__name__}: {e}")
            # diagnostic still reads what was captured before the raise

        if len(dfdraw_call_capture) == 0:
            pytest.fail(
                "K1_3: draw_batch produced no DFDraw calls. "
                "Cannot diagnose — possibly a different code path. "
                "Investigate draw_batch source."
            )

        report_lines = [
            f"K1_3 diagnostic — {len(dfdraw_call_capture)} DFDraw call(s):"
        ]
        per_call_drops = []
        for i, call in enumerate(dfdraw_call_capture):
            captured = call['kwargs']
            line = f"  call[{i}] DFDraw.{call['method']}() kwargs: "
            for key in ('group_by', 'group_by_bins', 'bins'):
                if key in captured:
                    line += f"{key}={captured[key]!r} "
                else:
                    line += f"{key}=ABSENT "
                    per_call_drops.append(f"call[{i}] missing {key}")
            report_lines.append(line)

        print("\n" + "\n".join(report_lines))

        assert not per_call_drops, (
            "K1_3 BUG LOCALIZATION: draw_batch did not forward all "
            f"batch-level kwargs to each inner call: {per_call_drops}\n"
            + "\n".join(report_lines)
        )

    @pytest.mark.invariance
    def test_K1_4_draw_figures_forwards_figure_kwargs(
        self, dfdraw_call_capture
    ):
        """
        K1_4 FIGURES DISPATCH.

        Same shape as K1_3 but for draw_figures. Each subplot must
        receive figure-level kwargs.
        """
        adf = _build_small_adf()

        # Simplest figure spec: one figure with 2 subplots
        specs = [
            {
                'name': 'fig1',
                'subplots': [
                    {'expr': 'y1:x', 'type': 'scatter'},
                    {'expr': 'y2:x', 'type': 'scatter'},
                ],
            },
        ]
        try:
            adf.draw_figures(
                specs, group_by='z', group_by_bins=4, bins=10
            )
        except Exception as e:
            print(f"\nK1_4 note: draw_figures raised "
                  f"{type(e).__name__}: {e}")

        if len(dfdraw_call_capture) == 0:
            pytest.skip(
                "K1_4: draw_figures produced no DFDraw calls — "
                "spec format may differ from assumption. Investigate "
                "before re-enabling."
            )

        report_lines = [
            f"K1_4 diagnostic — {len(dfdraw_call_capture)} DFDraw call(s):"
        ]
        per_call_drops = []
        for i, call in enumerate(dfdraw_call_capture):
            captured = call['kwargs']
            line = f"  call[{i}] DFDraw.{call['method']}() kwargs: "
            for key in ('group_by', 'group_by_bins', 'bins'):
                if key in captured:
                    line += f"{key}={captured[key]!r} "
                else:
                    line += f"{key}=ABSENT "
                    per_call_drops.append(f"call[{i}] missing {key}")
            report_lines.append(line)

        print("\n" + "\n".join(report_lines))

        assert not per_call_drops, (
            "K1_4 BUG LOCALIZATION: draw_figures did not forward all "
            f"figure-level kwargs to each subplot: {per_call_drops}\n"
            + "\n".join(report_lines)
        )

    @pytest.mark.invariance
    def test_K1_5_vector_expression_each_call_gets_full_kwargs(
        self, dfdraw_call_capture
    ):
        """
        K1_5 VECTOR EXPRESSION (the original bug reproducer).

        adf.draw("[y1,y2]:x", group_by='z', group_by_bins=6, bins=12)

        Behavior to diagnose:
        (a) ADF passes "[y1,y2]:x" as-is to DFDraw → 1 captured call.
            Vector expansion is dfdraw's responsibility.
        (b) ADF expands and dispatches → 2 captured calls.

        Either way: each captured call must have the full kwarg set.
        If a captured call is missing group_by / group_by_bins / bins,
        that is the bug, regardless of which layer expands.
        """
        adf = _build_small_adf()

        try:
            adf.draw(
                "[y1,y2]:x",
                group_by='z',
                group_by_bins=6,
                bins=12,
            )
        except Exception as e:
            print(f"\nK1_5 note: draw raised "
                  f"{type(e).__name__}: {e}")

        n_calls = len(dfdraw_call_capture)

        report_lines = [
            f"K1_5 diagnostic — vector expr '[y1,y2]:x' produced "
            f"{n_calls} DFDraw call(s):"
        ]
        per_call_drops = []
        if n_calls == 0:
            report_lines.append(
                "  WARNING: zero DFDraw calls — likely raised before "
                "reaching DFDraw. Investigate ADF-side vector handling."
            )
        for i, call in enumerate(dfdraw_call_capture):
            captured = call['kwargs']
            args = call['args']
            line = (
                f"  call[{i}] DFDraw.{call['method']}"
                f"(args={args!r}) kwargs: "
            )
            for key in ('group_by', 'group_by_bins', 'bins'):
                if key in captured:
                    line += f"{key}={captured[key]!r} "
                else:
                    line += f"{key}=ABSENT "
                    per_call_drops.append(f"call[{i}] missing {key}")
            report_lines.append(line)

        print("\n" + "\n".join(report_lines))

        assert n_calls > 0, (
            "K1_5: vector expr did not reach DFDraw at all.\n"
            + "\n".join(report_lines)
        )
        assert not per_call_drops, (
            "K1_5 BUG LOCALIZATION (the original report): "
            f"vector dispatch dropped kwargs: {per_call_drops}\n"
            + "\n".join(report_lines)
        )
