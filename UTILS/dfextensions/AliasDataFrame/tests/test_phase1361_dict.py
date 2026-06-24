"""
Phase 13.61.ADF — D-ADF-DICT test suite (PHASE document §8).

Verifies that draw()/draw_batch()/draw_figures() hand dfdraw a small dict-built
dispatch frame (only the needed columns) instead of the full-width frame, while
producing byte-identical results.

The implementation carries a `draw_dict` switch (default True). Tests run the SAME
public draw with the switch ON (dict) and OFF (full frame, == pre-fix behaviour) and
assert (a) identical numeric results and (b) the dict frame excludes decoy columns.
This is the FM#12 contract: the structural assertion FAILS with the switch off
(decoys present) and PASSES with it on.

Self-contained: builds synthetic AliasDataFrame frames, no ROOT / no .root file.
Gates: structural (primary, deterministic), AC-1 / AC-1a / AC-1b (C-0 equivalence),
negative control, block-count (no fragmentation), volume-invariance (alma2 BLOCKING),
peak-RSS (subprocess, peak not delta).
"""

import os
import sys
import subprocess
import contextlib
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dfextensions.AliasDataFrame import AliasDataFrame
from dfextensions.dfdraw.drawer import DFDraw


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_adf(n=20000, n_decoys=5, seed=0):
    """Synthetic ADF: a few real channels + an alias + a weights alias + decoys.

    Real channels: x, y, g (int group), gc (categorical), ts (datetime), sel.
    Alias 'sect' = '2*x' (materialized -> a real column dfdraw evals by name).
    Weights alias 'w' = '1.0 + abs(y)'.
    Decoys decoy0..decoy{n-1}: present in the frame, used by NO draw.
    """
    rng = np.random.default_rng(seed)
    data = {
        "x": rng.normal(0, 1, n).astype(np.float64),
        "y": rng.normal(0, 1, n).astype(np.float64),
        "g": rng.integers(0, 4, n).astype(np.int32),
        "sel": rng.normal(0, 1, n).astype(np.float64),
        "ts": (np.datetime64("2026-01-01")
               + rng.integers(0, 10_000_000, n).astype("timedelta64[s]")),
    }
    for i in range(n_decoys):
        data[f"decoy{i}"] = rng.normal(0, 1, n).astype(np.float64)
    df = pd.DataFrame(data)
    df["gc"] = pd.Categorical(
        rng.choice(["A", "B", "C"], size=n), categories=["A", "B", "C"], ordered=True
    )
    adf = AliasDataFrame(df)
    adf.draw_lazy = True
    adf.add_alias("sect", "2*x", dtype=np.float64)
    adf.add_alias("w", "1.0 + abs(y)", dtype=np.float64)
    adf.materialize_aliases(names=["sect", "w"])
    return adf


@contextlib.contextmanager
def _capture_dispatch_columns():
    """Capture the column set of every DataFrame DFDraw is constructed from."""
    captured = []
    orig = DFDraw.__init__

    def spy(self, data):
        if isinstance(data, pd.DataFrame):
            captured.append(set(data.columns))
        return orig(self, data)

    DFDraw.__init__ = spy
    try:
        yield captured
    finally:
        DFDraw.__init__ = orig
        plt.close("all")


def _stats_equal(a, b, rtol=0, atol=0):
    """Recursively assert two stats structures have identical numeric content."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), "stats dict keys differ"
        for k in a:
            _stats_equal(a[k], b[k], rtol, atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            _stats_equal(x, y, rtol, atol)
    else:
        try:
            aa = np.asarray(a, dtype=float)
            bb = np.asarray(b, dtype=float)
        except (TypeError, ValueError):
            return  # non-numeric leaf
        if aa.shape == bb.shape and aa.size:
            assert np.allclose(aa, bb, rtol=rtol, atol=atol, equal_nan=True), \
                "numeric stats differ between dict and full-frame"


def _draw_both(adf, **kw):
    """Run the same draw with dict ON and OFF, return (stats_on, stats_off)."""
    adf.draw_dict = True
    r_on = adf.draw(return_data=True, **kw)
    adf.draw_dict = False
    r_off = adf.draw(return_data=True, **kw)
    adf.draw_dict = True
    plt.close("all")
    s_on = r_on[2] if isinstance(r_on, (list, tuple)) and len(r_on) > 2 else {}
    s_off = r_off[2] if isinstance(r_off, (list, tuple)) and len(r_off) > 2 else {}
    return s_on, s_off


# ─────────────────────────────────────────────────────────────────────────────
# Structural gate (primary, deterministic, scale-invariant)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.invariance
def test_resolver_failure_warns_and_not_fullframe(monkeypatch):
    """P1-4: if get_required_branches raises, the dict must NOT silently revert to
    the full frame — it warns loudly and degrades to the token-scan (still projected)."""
    import warnings
    adf = _make_adf(n_decoys=5)
    adf.draw_dict = True

    def boom(*a, **k):
        raise RuntimeError("resolver boom")
    monkeypatch.setattr(adf, "get_required_branches", boom)

    with _capture_dispatch_columns() as cols:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            adf.draw("y:x", selection="sel>0", type="profile", bins=10)
    got = cols[-1]
    assert not [c for c in got if c.startswith("decoy")], \
        "resolver-failure fallback leaked decoys (reverted to full frame)"
    assert {"x", "y", "sel"} <= got, "token-scan degrade dropped referenced columns"
    assert any(issubclass(ww.category, RuntimeWarning) and "draw-dict" in str(ww.message)
               for ww in w), "resolver failure was silent (no loud warning)"


@pytest.mark.invariance
def test_decoy_named_like_tokens_not_overprojected():
    """P2-3: columns named like function tokens/literals are not projected unless
    actually referenced by a channel."""
    adf = _make_adf(n_decoys=0)
    for name in ["abs", "np", "int", "red", "blue"]:
        adf.df[name] = 1.0
    adf.draw_dict = True
    with _capture_dispatch_columns() as cols:
        adf.draw("y:x", type="profile", bins=10, group_by="g")
    got = cols[-1]
    leaked = {"abs", "np", "int", "red", "blue"} & got
    assert not leaked, f"token-named decoys over-projected: {leaked}"


@pytest.mark.invariance
def test_color_list_values_not_projected():
    """P2-2/GPT14: a color list (colour values, not column names) must not project
    columns even if same-named columns exist."""
    adf = _make_adf(n_decoys=0)
    adf.df["red"] = 1.0
    adf.df["blue"] = 2.0
    adf.draw_dict = True
    with _capture_dispatch_columns() as cols:
        adf.draw("y:x", type="profile", bins=10, color=["red", "blue"])
    got = cols[-1]
    assert "red" not in got and "blue" not in got, "color-list values leaked as columns"


@pytest.mark.invariance
def test_structural_dict_drops_decoys():
    """draw() dispatch frame must contain ONLY needed columns — no decoys."""
    adf = _make_adf(n_decoys=5)
    adf.draw_dict = True
    with _capture_dispatch_columns() as cols:
        adf.draw("y:sect", selection="sel>0", type="profile", bins=20, group_by="g")
    assert cols, "DFDraw was never constructed from a DataFrame"
    got = cols[-1]
    # needed: y, sect (alias name dfdraw evals), sel (selection), g (group_by)
    assert {"y", "sect", "sel", "g"} <= got
    decoys = {c for c in got if c.startswith("decoy")}
    assert not decoys, f"dispatch frame leaked decoy columns: {decoys}"


@pytest.mark.invariance
def test_structural_negative_control_full_frame_keeps_decoys():
    """Negative control (B2): with the dict OFF the frame DOES carry decoys.

    Proves the structural assertion above is not vacuous — it fails without the fix.
    """
    adf = _make_adf(n_decoys=5)
    adf.draw_dict = False
    with _capture_dispatch_columns() as cols:
        adf.draw("y:sect", selection="sel>0", type="profile", bins=20, group_by="g")
    got = cols[-1]
    decoys = {c for c in got if c.startswith("decoy")}
    assert decoys, "expected full frame (dict off) to still contain decoys"


@contextlib.contextmanager
def _capture_dispatch_frames():
    """Capture each DataFrame DFDraw is constructed from (column-subset copy)."""
    frames = []
    orig = DFDraw.__init__

    def spy(self, data):
        if isinstance(data, pd.DataFrame):
            frames.append(data.copy())
        return orig(self, data)

    DFDraw.__init__ = spy
    try:
        yield frames
    finally:
        DFDraw.__init__ = orig
        plt.close("all")


def _assert_dispatch_equiv(adf, **kw):
    """C-0 at the handoff: the frame dfdraw gets with the dict ON must be a
    column-subset of the full frame and numerically identical on every column
    it carries (and must drop decoys). Type-agnostic — no return_data needed."""
    adf.draw_dict = True
    with _capture_dispatch_frames() as fon:
        adf.draw(**kw)
    adf.draw_dict = False
    with _capture_dispatch_frames() as foff:
        adf.draw(**kw)
    adf.draw_dict = True
    assert fon and foff, "DFDraw not constructed from a DataFrame"
    don, doff = fon[-1], foff[-1]
    assert set(don.columns) <= set(doff.columns), "dict frame carries extra columns"
    assert not [c for c in don.columns if c.startswith("decoy")], "decoys leaked"
    for c in don.columns:
        a = don[c].reset_index(drop=True)
        b = doff[c].reset_index(drop=True)
        if pd.api.types.is_numeric_dtype(a):
            assert np.allclose(np.asarray(a, dtype=float),
                               np.asarray(b, dtype=float), equal_nan=True), \
                f"column {c!r} differs between dict and full frame"
        else:
            assert a.equals(b), f"column {c!r} differs between dict and full frame"


# ─────────────────────────────────────────────────────────────────────────────
# AC-1 — C-0 equivalence dict == full frame (incl. categorical + datetime)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.invariance
@pytest.mark.parametrize("kw", [
    dict(expr="y", type="hist", bins=30, selection="sel>0"),
    dict(expr="y:x", type="profile", bins=25, selection="sel>0"),
    dict(expr="y:x", type="profile", bins=25, group_by="g"),
    dict(expr="y:x", type="profile", bins=25, group_by="gc"),          # categorical
    dict(expr="y:x", type="profile", bins=25, facet_by="gc"),          # categorical facet
    dict(expr="y", type="hist", bins=30, weights="w"),                 # weights alias
    dict(expr="y:x", type="profile", bins=25,
         selection_vector=["sel>0", "sel<=0"], normalize="delta"),     # vector ch
])
def test_ac1_dict_equals_full_frame(kw):
    adf = _make_adf()
    _assert_dispatch_equiv(adf, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# AC-1a — subframe C-0 (single-level merge + col-in-index / rename / dedup fixes)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.invariance
def test_ac1a_subframe_dict_equals_full_frame():
    adf = _make_adf()
    adf.df["gidx"] = adf.df["g"].astype(np.int32)
    sf = AliasDataFrame(pd.DataFrame({
        "gidx": [0, 1, 2, 3],
        "meanY": [10.0, 20.0, 30.0, 40.0],
        # duplicate index entry to exercise drop_duplicates dedup fix:
    }))
    adf.register_subframe("GB", sf, index_columns=["gidx"])
    s_on, s_off = _draw_both(adf, expr="GB.meanY:gidx", type="profile", bins=4)
    _stats_equal(s_on, s_off)

    # col-in-index guard: subframe column IS the join key
    sf2 = AliasDataFrame(pd.DataFrame({"gidx": [0, 1, 2, 3]}))
    adf.register_subframe("GB2", sf2, index_columns=["gidx"])
    s_on2, s_off2 = _draw_both(adf, expr="GB2.gidx:g", type="profile", bins=4)
    _stats_equal(s_on2, s_off2)


# ─────────────────────────────────────────────────────────────────────────────
# AC-1b — batch C-0: draw_batch + draw_figures, specs needing DIFFERENT columns
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.invariance
def test_ac1b_batch_subframe_dict_equals_full_frame():
    """draw_batch with a subframe ref in a spec — dict on/off identical (Opus48_1 §8).

    NOTE: draw_batch mutates the passed specs dict (rewrites 'GB.meanY'->'GB_meanY'),
    a pre-existing behaviour unrelated to the dict; we pass a fresh deepcopy per call.
    """
    import copy
    adf = _make_adf()
    adf.df["gidx"] = adf.df["g"].astype(np.int32)
    sf = AliasDataFrame(pd.DataFrame({"gidx": [0, 1, 2, 3],
                                      "meanY": [10.0, 20.0, 30.0, 40.0]}))
    adf.register_subframe("GB", sf, index_columns=["gidx"])
    specs = {
        "p_plain": {"expr": "y:x", "type": "profile", "bins": 20},
        "p_sf":    {"expr": "GB.meanY:gidx", "type": "profile", "bins": 4},
    }
    adf.draw_dict = True
    res_on = adf.draw_batch(copy.deepcopy(specs), verbose=False)
    adf.draw_dict = False
    res_off = adf.draw_batch(copy.deepcopy(specs), verbose=False)
    adf.draw_dict = True
    plt.close("all")
    for name in specs:
        _stats_equal(res_on[name].get("stats", {}), res_off[name].get("stats", {}))


@pytest.mark.invariance
def test_ac1b_batch_dict_equals_full_frame():
    adf = _make_adf()
    specs = {
        "p_yx": {"expr": "y:x", "type": "profile", "bins": 20},
        "p_yg": {"expr": "y:sect", "type": "profile", "bins": 20, "group_by": "g"},
        "h_y":  {"expr": "y", "type": "hist", "bins": 20, "selection": "sel>0"},
    }
    adf.draw_dict = True
    res_on = adf.draw_batch(specs, verbose=False)
    adf.draw_dict = False
    res_off = adf.draw_batch(specs, verbose=False)
    adf.draw_dict = True
    plt.close("all")
    for name in specs:
        _stats_equal(res_on[name].get("stats", {}), res_off[name].get("stats", {}))


@pytest.mark.invariance
def test_ac1b_figures_dict_equals_full_frame():
    adf = _make_adf()
    figs = [{
        "name": "dash",
        "plots": [
            {"expr": "y:x", "type": "profile", "bins": 20},
            {"expr": "y", "type": "hist", "bins": 20, "selection": "sel>0"},
        ],
    }]
    adf.draw_dict = True
    res_on = adf.draw_figures(figs, verbose=False)
    adf.draw_dict = False
    res_off = adf.draw_figures(figs, verbose=False)
    adf.draw_dict = True
    plt.close("all")
    _stats_equal(res_on["dash"].get("stats", {}), res_off["dash"].get("stats", {}))


# ─────────────────────────────────────────────────────────────────────────────
# Block-count — the big frame is never copied or grown (no fragmentation)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.invariance
def test_block_count_self_df_unchanged():
    adf = _make_adf()
    adf.draw_dict = True

    def nblocks():
        try:
            return len(adf.df._mgr.blocks)
        except Exception:
            return len(adf.df._data.blocks)

    cols_before = list(adf.df.columns)
    before = nblocks()
    for _ in range(10):
        adf.draw("y:sect", selection="sel>0", type="profile", bins=20, group_by="g")
    plt.close("all")
    assert list(adf.df.columns) == cols_before, "self.df columns changed after draws"
    assert nblocks() == before, "self.df block count changed (fragmentation)"


# ─────────────────────────────────────────────────────────────────────────────
# Volume-invariance — alma2 BLOCKING gate (>=1M rows, >=50 cols, >=50 decoys)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.volume_invariance
def test_volume_invariance_dispatch_excludes_decoys():
    adf = _make_adf(n=1_000_000, n_decoys=60)
    assert len([c for c in adf.df.columns if c.startswith("decoy")]) >= 50
    assert len(adf.df.columns) >= 50
    adf.draw_dict = True
    with _capture_dispatch_columns() as cols:
        adf.draw("y:sect", selection="sel>0", type="profile", bins=50, group_by="g")
    got = cols[-1]
    assert {"y", "sect", "sel", "g"} <= got
    assert not {c for c in got if c.startswith("decoy")}, "decoys leaked at volume"


# ─────────────────────────────────────────────────────────────────────────────
# Peak-RSS — subprocess-isolated, peak (ru_maxrss) not delta. Dict ON must hold
# materially less peak memory than the full-frame copy path on a large wide frame.
# Exercises a group_by-EXPRESSION (forces df_subset.copy() in the full-frame path).
# ─────────────────────────────────────────────────────────────────────────────

_PEAK_CHILD = r'''
import sys, resource
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
from dfextensions.AliasDataFrame import AliasDataFrame
flag = sys.argv[1] == "on"
n, ncol = 1_000_000, 60
rng = np.random.default_rng(0)
data = {"x": rng.normal(0,1,n), "y": rng.normal(0,1,n)}
for i in range(ncol):
    data[f"d{i}"] = rng.normal(0,1,n)
adf = AliasDataFrame(pd.DataFrame(data))
adf.draw_lazy = True
adf.draw_dict = flag
# group_by EXPRESSION -> full-frame path does df_subset.copy() of the whole frame
adf.draw("y:x", type="profile", bins=50, group_by="x>0.5")
peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
print(peak_kb)
'''


@pytest.mark.volume_invariance
def test_peak_rss_dict_below_full_frame():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    script = os.path.join(os.path.dirname(__file__), "_peak_child_1361.py")
    with open(script, "w") as f:
        f.write(_PEAK_CHILD)
    try:
        def peak(flag):
            out = subprocess.check_output([sys.executable, script, flag], env=env)
            return int(out.strip().splitlines()[-1])
        peak_on = peak("on")
        peak_off = peak("off")
    finally:
        with contextlib.suppress(OSError):
            os.remove(script)
    # full frame is ~1M x 62 float64 ~ 0.5 GB; the dict path must avoid duplicating it.
    assert peak_on < peak_off, f"dict peak {peak_on} !< full-frame peak {peak_off}"
    # and the saving should be substantial (at least ~200 MB == 200_000 KB)
    assert (peak_off - peak_on) > 200_000, \
        f"expected >200MB peak saving; got {(peak_off-peak_on)} KB"
