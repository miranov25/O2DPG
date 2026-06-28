"""PHASE 13.63 — Generalized Diff f(S): the agreed §14 invariance & smoke suite.

Each test names a §14 entry and states its invariant in the docstring. Two
confidence classes:
  * baseline-free (true by mathematics)  : B4, B5  -> a failure means wrong code.
  * equivalence / regression             : B1, B2, B3, A5.

Fixture discipline (the near-miss this suite must prevent): selections must
OVERLAP in x, or the N-curve mask empties every bin and invariants pass
*vacuously*. The fixture uses x-independent flags (isMC, isEdge) so all four
(MC/Data x Edge/NotEdge) combinations span every x-bin; `_valid()` asserts a
minimum populated-bin count so a vacuous pass cannot masquerade as real.

Order-lock (§4) is tested first, before any behavioural assertion.
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from dfdraw.drawer import DFDraw  # shim package; canonical drawer.py e851acc7

BINS = 24
MIN_ENTRIES = 50
MIN_VALID_BINS = 10          # guard against vacuous passes


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(20260628)
    n = 120_000
    x = rng.uniform(-2.0, 2.0, n)
    isMC = rng.integers(0, 2, n)           # flags INDEPENDENT of x -> overlap in x
    isEdge = rng.integers(0, 2, n)
    y = 1.0 + 0.10 * x + 0.02 * isMC + 0.01 * isEdge + rng.normal(0, 0.05, n)
    return pd.DataFrame({
        "x": x, "y": y, "isMC": isMC, "isEdge": isEdge,
        "sector": rng.integers(0, 2, n), "side": rng.integers(0, 2, n),
        # y-vector source columns (D-alpha typed-surface arm of B6/A4):
        "y_edge":    np.where(isEdge == 1, y, np.nan),
        "y_notEdge": np.where(isEdge == 0, y, np.nan),
        # binary weights (B6 weights arm: weighted-mean == selected-mean):
        "w_edge":    (isEdge == 1).astype(float),
        "w_notEdge": (isEdge == 0).astype(float),
    })


S2 = ["isEdge==1", "isEdge==0"]
S4 = ["isMC==0 & isEdge==1", "isMC==0 & isEdge==0",
      "isMC==1 & isEdge==1", "isMC==1 & isEdge==0"]


# ---------- helpers ----------------------------------------------------------
def _nd(df, sel=None, mode=None, key="normalize_data", **kw):
    f, ax, s = DFDraw(df).draw("y:x", type="profile", selection_vector=sel,
                               vector_compose="outer", normalize=mode,
                               bins=BINS, min_entries=MIN_ENTRIES, **kw)
    return s[key]


def _valid(nd):
    """Valid-bin value array (min_entries NaN-mask filtered)."""
    v = nd[~nd["mask_undefined"]]
    assert len(v) >= MIN_VALID_BINS, f"vacuous: only {len(v)} valid bins"
    return v["value"].to_numpy()


# ---------- §4 ORDER-LOCK (first, before any behaviour change) ---------------
def test_order_lock_flattening():
    """§4: the flat order of S is frozen — y slowest, selection middle, weights
    fastest. The 4-selection deadline case maps S[0..3] in selection order."""
    f = DFDraw._compute_vector_iteration_indices
    assert f(1, S4, None, "outer") == [(0, 0, None), (0, 1, None),
                                       (0, 2, None), (0, 3, None)]
    assert f(2, ["a", "b"], None, "outer") == [(0, 0, None), (0, 1, None),
                                               (1, 0, None), (1, 1, None)]
    assert f(2, ["a", "b"], ["u", "v"], "outer") == [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)]


# ---------- PART A — must-not-fail (smoke) -----------------------------------
@pytest.mark.parametrize("mode", ["ratio", "delta", "log_ratio", "pull"])
def test_A1_string_modes_two_curve(df, mode):
    """A1: each built-in string mode runs on a 2-curve selection source."""
    _nd(df, S2, mode)


@pytest.mark.xfail(reason="multi-output f(S) deferred to §9.4/B11; "
                          "single-output covers the deadline (raises NotImplementedError)",
                   strict=True)
def test_A2_n_ratio_multi_output(df):
    """A2: N-ratio via a multi-output callable. Multi-output (a list of N-1
    derived curves) is a named follow-on; the implementation raises until
    §9.4 lands. Value-correctness of N-ratio is covered single-output in B3."""
    _nd(df, S4, lambda S: [S[k] / S[-1] for k in range(len(S) - 1)])


def test_A3_double_ratio(df):
    """A3: the deadline double ratio runs in one call."""
    _nd(df, S4, lambda S: (S[0] / S[1]) / (S[2] / S[3]))


def test_A4_weights_source(df):
    """A4: callable on the weights_vector source."""
    f, ax, s = DFDraw(df).draw("y:x", type="profile",
                               weights_vector=["w_edge", "w_notEdge"],
                               vector_compose="outer",
                               normalize=lambda S: S[0] / S[1],
                               bins=BINS, min_entries=MIN_ENTRIES)
    assert "normalize_data" in s


def test_A5_three_dispatchers(df):
    """A5 (PF-1): the callable must run on plain, grouped, and faceted
    dispatchers — otherwise normalize=callable diverges with/without
    group_by/facet_by (the silent-correctness bug PF-1 guards). Each
    dispatcher returns its own stats key."""
    f = lambda S: S[0] / S[1]
    _, _, sp = DFDraw(df).draw("y:x", type="profile", selection_vector=S2,
                               vector_compose="outer", normalize=f,
                               bins=BINS, min_entries=MIN_ENTRIES)
    assert "normalize_data" in sp                                  # plain   :2619
    _, _, sg = DFDraw(df).draw("y:x", type="profile", selection_vector=S2,
                               vector_compose="outer", normalize=f,
                               group_by="sector", bins=BINS, min_entries=MIN_ENTRIES)
    assert "normalize_data_grouped" in sg                          # grouped :2935
    _, _, sf = DFDraw(df).draw("y:x", type="profile", selection_vector=S2,
                               vector_compose="outer", normalize=f,
                               facet_by="side", bins=BINS, min_entries=MIN_ENTRIES)
    assert "normalize_data_faceted" in sf                          # faceted :3265


def test_A6_entry_gate_widened(df):
    """A6: a length-4 selection_vector must NOT raise at the entry gate
    (drawer.py:6140, widened !=2 -> <2)."""
    _nd(df, S4, lambda S: (S[0] / S[1]) / (S[2] / S[3]))


# ---------- PART B — correctness by invariance -------------------------------
def test_B1_g0_string_ratio_runs(df):
    """B1 (G-0): the 2-curve string 'ratio' produces finite output on populated
    bins. (Byte-identity vs a saved pre-change baseline is asserted in the
    cross-version harness; here we assert the path is intact.)"""
    v = _valid(_nd(df, S2, "ratio"))
    assert np.all(np.isfinite(v))


@pytest.mark.parametrize("mode,lam", [
    ("ratio",     lambda S: S[0] / S[1]),
    ("delta",     lambda S: S[0] - S[1]),
    ("log_ratio", lambda S: np.log(S[0] / S[1])),
])
def test_B2_preset_equals_lambda(df, mode, lam):
    """B2: each string mode's VALUES equal the explicit-lambda values (the
    generalization did not perturb the presets). Errors differ by design —
    string modes carry analytic errors; the bare callable defers errors to
    f_err (§5) — so this asserts values only."""
    vm = _valid(_nd(df, S2, mode))
    vl = _valid(_nd(df, S2, lam))
    assert np.allclose(vm, vl, equal_nan=True)


def test_B3_ac2_single_output(df):
    """B3 (AC-2 — the §2A.5 guard): f=S[0]/S[1] is byte-identical to the string
    'ratio' of the same pair. Single-output, so no dependence on the multi-
    output stats key. Asserted for selection AND weights sources."""
    sig, ref = "isMC==0 & isEdge==1", "isMC==1 & isEdge==0"
    n = _valid(_nd(df, [sig, ref], lambda S: S[0] / S[1]))
    p = _valid(_nd(df, [sig, ref], "ratio"))
    assert np.array_equal(n, p, equal_nan=True)
    # weights-sourced pair (same split via binary weights):
    df2 = df.copy()
    df2["w_sig"] = df.eval(sig).astype(float)
    df2["w_ref"] = df.eval(ref).astype(float)
    nw = DFDraw(df2).draw("y:x", type="profile",
                          weights_vector=["w_sig", "w_ref"], vector_compose="outer",
                          normalize=lambda S: S[0] / S[1], bins=BINS, min_entries=MIN_ENTRIES)[2]["normalize_data"]
    pw = DFDraw(df2).draw("y:x", type="profile",
                          weights_vector=["w_sig", "w_ref"], vector_compose="outer",
                          normalize="ratio", bins=BINS, min_entries=MIN_ENTRIES)[2]["normalize_data"]
    m = ~nw["mask_undefined"].to_numpy() & ~pw["mask_undefined"].to_numpy()
    assert m.sum() >= MIN_VALID_BINS
    assert np.allclose(nw["value"].to_numpy()[m], pw["value"].to_numpy()[m])


def test_B4_double_ratio_equals_manual(df):
    """B4 (baseline-free): the double ratio equals the ratio of two
    independently-computed ratios — validates the whole f(S) machine against
    the manual pandas workflow it replaces."""
    dr = _nd(df, S4, lambda S: (S[0] / S[1]) / (S[2] / S[3]))
    rD = _nd(df, S4[:2], "ratio")
    rM = _nd(df, S4[2:], "ratio")
    m = (~dr["mask_undefined"].to_numpy()
         & ~rD["mask_undefined"].to_numpy()
         & ~rM["mask_undefined"].to_numpy())
    assert m.sum() >= MIN_VALID_BINS
    assert np.allclose(dr["value"].to_numpy()[m],
                       (rD["value"].to_numpy() / rM["value"].to_numpy())[m])


def test_B5_algebraic_identities(df):
    """B5 (baseline-free): identities that must hold by mathematics."""
    assert np.allclose(_valid(_nd(df, ["isEdge==1", "isEdge==1"], "ratio")), 1.0)
    assert np.allclose(_valid(_nd(df, ["isEdge==1", "isEdge==1"], "delta")), 0.0)
    assert np.allclose(_valid(_nd(df, ["isEdge==1", "isEdge==0"],
                                  lambda S: (S[0] / S[1]) * (S[1] / S[0]))), 1.0)
    assert np.allclose(_valid(_nd(df, ["isEdge==1", "isEdge==1",
                                       "isEdge==0", "isEdge==0"],
                                  lambda S: (S[0] / S[1]) / (S[2] / S[3]))), 1.0)


def test_B6_source_independence(df):
    """B6: the same Edge/NotEdge split via selection, y-vector (typed profile()
    surface, D-alpha), and binary weights gives identical ratio curves —
    confirming the operand is a vector-iteration curve, not a syntax form.
    Cross-channel comparison uses allclose (FP summation-order, per v1.2)."""
    a = _nd(df, ["isEdge==1", "isEdge==0"], "ratio")
    c = DFDraw(df).draw("y:x", type="profile",
                        weights_vector=["w_edge", "w_notEdge"], vector_compose="outer",
                        normalize="ratio", bins=BINS, min_entries=MIN_ENTRIES)[2]["normalize_data"]
    # y-vector typed surface (D-alpha): profile(), not draw()
    fb, axb, sb = DFDraw(df).profile("[y_edge, y_notEdge]:x",
                                     normalize="ratio", bins=BINS, min_entries=MIN_ENTRIES)
    b = sb["normalize_data"]
    m = (~a["mask_undefined"].to_numpy()
         & ~b["mask_undefined"].to_numpy()
         & ~c["mask_undefined"].to_numpy())
    assert m.sum() >= MIN_VALID_BINS
    assert np.allclose(a["value"].to_numpy()[m], c["value"].to_numpy()[m])  # selection == weights
    assert np.allclose(a["value"].to_numpy()[m], b["value"].to_numpy()[m])  # selection == y-vector


def test_B7_group_by_orthogonality(df):
    """B7 (§6b): f applied per group equals looping the groups by hand (direct
    frame subset — adf.query does not exist, F-3). The grouped payload is a dict
    keyed by group value; each entry is a dict with 'values'/'mask_undefined'
    arrays.

    Tolerance note (verified): the grouped path bins every group on the GLOBAL
    x_range (shared across groups — consistent with the §7 alignment guarantee),
    while a manual subset autoranges to its own x-extent, shifting bin edges by
    ~1e-4. The public API does not let x_range be pinned through the grouped
    path, so we assert agreement to rtol=1e-3 (values match to ~5 significant
    figures). A real orthogonality bug (wrong group's data, or 2*n_groups
    curves) would be order-1, not 1e-4, so this still catches structural breaks.
    """
    G = _nd(df, S2, lambda S: S[0] / S[1], group_by="sector",
            key="normalize_data_grouped")
    assert len(G) == df["sector"].nunique()           # f called once per group
    for sval, gnd in G.items():
        sub = DFDraw(df[df["sector"] == int(sval)])
        ref = sub.draw("y:x", type="profile", selection_vector=S2,
                       vector_compose="outer", normalize=lambda S: S[0] / S[1],
                       bins=BINS, min_entries=MIN_ENTRIES)[2]["normalize_data"]
        g_mask = np.asarray(gnd["mask_undefined"])
        g_val = np.asarray(gnd["values"])
        m = ~g_mask & ~ref["mask_undefined"].to_numpy()
        assert m.sum() >= MIN_VALID_BINS
        assert np.allclose(g_val[m], ref["value"].to_numpy()[m], rtol=1e-3)


def test_B8_trap_guard_raises(df):
    """B8: a column-naming reference (normalize_ref=) cannot address a
    vector-sourced operand and must be rejected (§2A.5 trap guard)."""
    with pytest.raises(Exception):
        DFDraw(df).draw("y:x", type="profile", selection_vector=["isEdge==1", "isEdge==0"],
                        normalize_ref="isEdge==0", bins=BINS, min_entries=MIN_ENTRIES)


def test_B9_error_survival_analytic_path(df):
    """B9: errors are not silently dropped on the path that HAS errors. The
    string 'ratio' mode propagates analytic errors; assert they are present,
    finite, and positive on populated bins. (The bare-callable path returns
    error=None by design, §5 — f_err is the user's hook; B9 therefore tests
    the analytic surface. Flagged for CRR: v1.2 B9 literal form tested the
    lambda path, which conflicts with §5.)"""
    nd = _nd(df, S2, "ratio")
    e = nd[~nd["mask_undefined"]]["error"].to_numpy()
    assert len(e) >= MIN_VALID_BINS
    assert np.all(np.isfinite(e)) and np.all(e > 0)


def test_B10_occupancy_desync(df):
    """B10: disjoint per-curve occupancy must not shift elementwise arithmetic.
    Two selections with disjoint x-support share the edge grid; bins populated
    in only one curve are NaN-masked (not paired with a neighbour), so no index
    shift. We assert the surviving bins are correctly masked, not misaligned."""
    nd = _nd(df, ["x<-1.0", "x>1.0"], "ratio")   # disjoint x-support
    # Every bin should be masked (no bin has BOTH selections populated):
    # the result must be all-undefined, never a spurious finite value from a
    # shifted pairing.
    assert bool(nd["mask_undefined"].all()), \
        "disjoint occupancy produced a finite bin -> index shift / misalignment"
