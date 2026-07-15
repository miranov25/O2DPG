"""PHASE_13_76_DF locking tests (v1.1) — facet float16 crash + 2D bin-edge recompute.

v1.0 REGRESSION (panel-caught, GPT12/15/16/17): the original Bug-B tests were
tautological — they compared df[mask] against mask.sum() (always equal) and never
invoked _dispatch_2d_facet, so a mutation reverting the fix still passed them.
This v1.1 replaces them with tests that call the PUBLIC facet dispatch path and
compare the dispatcher's returned per-cell stats against an INDEPENDENT oracle
computed outside the dispatcher. Mutation-verified: the B-tests FAIL when
_dispatch_2d_facet is reverted to the chained filter.

Bug A (P2): pd.cut/pd.qcut crash on float16 facet columns.
Bug B (P1): 2D facet chained the per-cell filter, re-deriving column bin edges on
            the already-row-filtered subset; populated cells silently under-reported.

Layer classification (panel correction 3):
  - B1/B2  -> "invariance" (dispatcher output vs independent oracle; the real lock)
  - A1/A2  -> "integration" (public dispatch path, float16, no-crash + grid shape)
  - structural/fixture checks -> "smoke"
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from dfdraw import DFDraw
from dfdraw.drawer import _safe_cut_series, _classify_facet_dim


def _correlated_frame(n=20000, seed=0):
    """Correlated columns: the column's range genuinely shifts per row-slice — the
    condition the existing 23 FBY tests never create (they use independent uniforms),
    which is exactly why they never caught Bug B."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 3.0 * x + rng.normal(0, 0.05, n)
    return pd.DataFrame({"row": x, "col": y})


def _float16_frame(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"row": rng.uniform(-1, 1, n).astype(np.float16),
                         "col": rng.uniform(-1, 1, n).astype(np.float16)})


# ============================ Bug B — REAL dispatcher locks =================
def _dispatch_cell_counts(df, kw):
    """Call the PUBLIC 2D facet path; return {(row_iv,col_iv): n} from the
    dispatcher's own returned stats dict."""
    fig, axes, stats = DFDraw(df).profile("col:row", facet_by=["row", "col"], **kw)
    import matplotlib.pyplot as plt; plt.close(fig)
    return {k: v.get("n", 0) for k, v in stats.items()}


def _oracle_cell_counts(df, mode):
    """Independent ground truth: GLOBAL right-closed cut/qcut, computed WITHOUT
    going through the dispatcher, counted per (row_bin, col_bin) cell."""
    if mode == "bins":
        rb, cb = pd.cut(df["row"], 3), pd.cut(df["col"], 3)
    else:
        rb = pd.qcut(df["row"], 3, duplicates="drop")
        cb = pd.qcut(df["col"], 3, duplicates="drop")
    return rb, cb


def test_FBY16_B1_dispatch_cell_counts_match_oracle_bins():
    """Bug B invariant (pd.cut): the dispatcher's per-cell n must equal an
    independent global-cut oracle, and total rows must be conserved. FAILS on the
    pre-fix chained-filter dispatcher (mutation-verified)."""
    df = _correlated_frame()
    disp = _dispatch_cell_counts(df, dict(facet_by_bins=[3, 3]))
    rb, cb = _oracle_cell_counts(df, "bins")
    total = 0
    for (riv, civ), n in disp.items():
        truth = int(((rb == riv) & (cb == civ)).sum())
        assert n == truth, f"cell ({riv},{civ}): dispatcher n={n} != oracle {truth}"
        total += n
    assert total == len(df), f"2D facet lost rows: dispatcher kept {total} of {len(df)}"


def test_FBY16_B2_dispatch_cell_counts_match_oracle_quantiles():
    """Bug B invariant (pd.qcut), same dispatcher-vs-oracle comparison."""
    df = _correlated_frame()
    disp = _dispatch_cell_counts(df, dict(facet_by_quantiles=[3, 3]))
    rb, cb = _oracle_cell_counts(df, "quantiles")
    total = 0
    for (riv, civ), n in disp.items():
        truth = int(((rb == riv) & (cb == civ)).sum())
        assert n == truth, f"cell ({riv},{civ}): dispatcher n={n} != oracle {truth}"
        total += n
    assert total == len(df), f"2D facet lost rows: dispatcher kept {total} of {len(df)}"


# ============================ Bug A — REAL public-path integration ==========
def test_FBY16_A1_float16_public_dispatch_bins():
    """Bug A (pd.cut): faceting on a float16 column through the PUBLIC dispatch
    path must not raise and must produce the expected grid shape."""
    df = _float16_frame()
    fig, axes, stats = DFDraw(df).profile("col:row", facet_by=["row", "col"],
                                          facet_by_bins=[3, 3])
    import matplotlib.pyplot as plt; plt.close(fig)
    assert len(stats) >= 1, "float16 facet produced no cells"
    assert np.asarray(axes).size >= 4, "expected a >=3x3-ish grid"


def test_FBY16_A2_float16_public_dispatch_quantiles():
    """Bug A (pd.qcut): float16 through the public path, quantile binning."""
    df = _float16_frame()
    fig, axes, stats = DFDraw(df).profile("col:row", facet_by=["row", "col"],
                                          facet_by_quantiles=[3, 3])
    import matplotlib.pyplot as plt; plt.close(fig)
    assert len(stats) >= 1, "float16 qcut facet produced no cells"


# ============================ structural / fixture (smoke) ==================
def test_FBY16_S1_float16_upcast_preserves_binning():
    """Structural: the float16->float32 upcast must not change the classification."""
    raw = np.random.default_rng(1).uniform(-1, 1, 2000).astype(np.float16)
    c16 = _safe_cut_series(pd.Series(raw), bins=4)
    c32 = _safe_cut_series(pd.Series(raw.astype(np.float32)), bins=4)
    assert (c16.astype(str) == c32.astype(str)).all()


def test_FBY16_S2_qcut_duplicates_drop_preserved():
    """Structural: qcut path keeps duplicates='drop' (heavy ties must not raise)."""
    s = pd.Series([0.0] * 900 + list(np.linspace(0.1, 1.0, 100)))
    assert _safe_cut_series(s, quantiles=4).notna().sum() > 0


def test_FBY16_S3_discrete_facet_unaffected():
    """Structural: bins/quantiles both None -> plain equality, no classification
    (guards the `is not None` branch; a truthiness guard would misroute bins=0)."""
    df = pd.DataFrame({"row": [0, 0, 1, 1, 2, 2], "col": [0, 1, 0, 1, 0, 1]})
    bin_series, values = _classify_facet_dim(df, "row", bins=None, quantiles=None)
    assert bin_series is None and values == [0, 1, 2]


def test_FBY16_S4_correlated_fixture_is_nonvacuous():
    """Fixture-quality guard: the correlated frame must actually make the column
    range shift across row slices, else B1/B2 would not exercise the bug."""
    df = _correlated_frame()
    rb = pd.cut(df["row"], 3)
    spans = [df.loc[rb == iv, "col"].max() - df.loc[rb == iv, "col"].min()
             for iv in rb.cat.categories]
    full = df["col"].max() - df["col"].min()
    assert min(spans) < 0.75 * full, "fixture too weak to trigger the chained-recompute bug"
