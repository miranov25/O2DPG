"""PHASE_13_74_GB_Explain tests E1-E11 (proposal v1.5).

Proposal v1.6 conformance suite. Sandbox note (CRR §env): old-ADF leg predates PHASE_13_73, so the
`add_alias(source=)` primary path is exercised on alma2 only; here the
documented D-5 fallback (pre-qualified formulas) runs. E4's source= leg
carries a skipif on signature detection.
"""
import sys, os, warnings
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ADF_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "AliasDataFrame")
if os.path.isdir(ADF_DIR):
    sys.path.insert(0, ADF_DIR)
sys.path.insert(0, os.environ.get("ADF_PATH", "/home/claude/work/adf"))  # sandbox layout; harmless on alma2

from AliasDataFrame import AliasDataFrame  # noqa: E402
import gb_explain  # noqa: E402
from gb_explain import (make_contribution_aliases, contribution_summary,  # noqa: E402
                        make_block_delta_aliases)

RNG = np.random.default_rng(1974)
HAS_SOURCE_KW = "source" in __import__("inspect").signature(
    AliasDataFrame.add_alias).parameters


def _meta(target="y", suffix="", nvars=("x1", "x2"), weights=None,
          fit_intercept=True):
    cols = [f"{target}_slope_{v}{suffix}" for v in nvars]
    if fit_intercept:
        cols.append(f"{target}_intercept{suffix}")
    m = {"columns": {"coefficients": {target: cols}},
         "parameters": {"suffix": suffix}}
    if weights:
        m["parameters"]["weights_column"] = weights
    return m


def _pergroup_frame(suffix="", fit_intercept=True, n=400):
    """Per-group fixture: two groups, DIFFERENT slopes per group (E1 rule);
    slope/intercept columns materialized into the frame (production
    post-join layout)."""
    g = np.repeat([0, 1], n // 2)
    x1 = RNG.normal(size=n)
    x2 = RNG.normal(size=n)
    b1 = np.where(g == 0, 1.5, -0.7)
    b2 = np.where(g == 0, 0.3, 2.1)
    b0 = np.where(g == 0, 5.0, -2.0) if fit_intercept else np.zeros(n)
    y = b0 + b1 * x1 + b2 * x2
    df = pd.DataFrame({"g": g, "x1": x1, "x2": x2, "y": y,
                       f"y_slope_x1{suffix}": b1, f"y_slope_x2{suffix}": b2})
    if fit_intercept:
        df[f"y_intercept{suffix}"] = b0
    return df


# ---------------------------------------------------------------- E1
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_e1_identity_baseline_plus_contributions_is_prediction(fit_intercept):
    suffix = "_v4"
    df = _pergroup_frame(suffix=suffix, fit_intercept=fit_intercept)
    adf = AliasDataFrame(df)
    meta = _meta(suffix=suffix, fit_intercept=fit_intercept)
    names = make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)
    reg = adf.df.attrs["gb_explain"]["y"]
    # float64-exact per §3a: evaluate the registered formulas... here the
    # identity in float64 from the same constants the aliases carry:
    c = np.zeros(len(df))
    for v, slope_col in reg["terms"].items():
        c += df[slope_col].to_numpy(np.float64) * (
            df[v].to_numpy(np.float64) - reg["mu"][v])
    base = np.zeros(len(df))
    for v, slope_col in reg["terms"].items():
        base += df[slope_col].to_numpy(np.float64) * reg["mu"][v]
    if fit_intercept:
        base += df[f"y_intercept{suffix}"].to_numpy(np.float64)
    pred = df["y"].to_numpy(np.float64)  # noiseless fixture: y == prediction
    np.testing.assert_allclose(base + c, pred, rtol=0, atol=1e-12)
    # and the stored aliases satisfy it at their dtype tolerance (float32)
    tot = None
    for n_ in names:
        adf.materialize_alias(n_) if hasattr(adf, "materialize_alias") else None
    ev = adf.df
    cols = [n_ for n_ in names]
    got = sum(ev[n_].to_numpy(np.float64) for n_ in cols)
    np.testing.assert_allclose(got, pred, rtol=1e-6, atol=1e-5)


# ---------------------------------------------------------------- E2 + E11 oracle
def _analytic_3term_cov(r, betas=(2.0, 1.0, 1.0)):
    """Population covariances of CONTRIBUTIONS c_j = beta_j x_j for
    standardized x1,x2 (corr r), x3 indep; y = sum c_j. Asymmetric betas
    on the correlated pair are REQUIRED: LMG == partial-sum shortcut
    exactly when the correlated contributions have equal stds (derived
    in-session 2026-07-13) — the symmetric fixture cannot discriminate."""
    b = np.asarray(betas, float)
    cov_x = np.array([[1.0, r, 0.0], [r, 1.0, 0.0], [0.0, 0.0, 1.0]])
    cov_cc = cov_x * np.outer(b, b)
    ones = np.ones(3)
    cov_cy = cov_cc @ ones
    var_y = float(ones @ cov_cc @ ones)
    return cov_cc, cov_cy, var_y


def _analytic_lmg(r, betas=(2.0, 1.0, 1.0)):
    """Hand-derived from the pinned §3a formula (subset-regression R2 from
    covariances), evaluated on explicitly written closed-form subset R2
    values — independent of the library implementation. betas=(2,1,1),
    corr(x1,x2)=r, x3 independent, y = c1+c2+c3:
    cov_cc = [[4, 2r, 0],[2r, 1, 0],[0, 0, 1]]; cov_cy = (4+2r, 1+2r, 1);
    vy = 6+4r.  R2({j}) = cov_cy_j^2 / (cov_cc_jj * vy).
    R2 of pairs/full from the same closed-form matrix algebra, written out."""
    vy = 6.0 + 4.0 * r
    c1y, c2y, c3y = 4.0 + 2.0 * r, 1.0 + 2.0 * r, 1.0
    r2_1 = c1y ** 2 / (4.0 * vy)
    r2_2 = c2y ** 2 / (1.0 * vy)
    r2_3 = 1.0 / vy
    # pair {1,2}: full 2x2 solve — explained = y minus x3 part = vy - 1
    r2_12 = (vy - 1.0) / vy
    r2_13 = r2_1 + r2_3
    r2_23 = r2_2 + r2_3
    r2_123 = 1.0
    orders = [(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1)]
    R2 = {frozenset(): 0.0, frozenset({1}): r2_1, frozenset({2}): r2_2,
          frozenset({3}): r2_3, frozenset({1, 2}): r2_12,
          frozenset({1, 3}): r2_13, frozenset({2, 3}): r2_23,
          frozenset({1, 2, 3}): r2_123}
    shares = {1: 0.0, 2: 0.0, 3: 0.0}
    for order in orders:
        seen = set()
        for j in order:
            shares[j] += R2[frozenset(seen | {j})] - R2[frozenset(seen)]
            seen.add(j)
    return np.array([shares[1], shares[2], shares[3]]) / 6.0


def _exact_moment_columns(n, cov, rng):
    """Data whose SAMPLE covariance (population form, /n) is EXACTLY cov."""
    k = cov.shape[0]
    X = rng.normal(size=(n, k))
    X -= X.mean(axis=0)
    # empirical whitening then coloring
    cs = np.linalg.cholesky(np.cov(X.T, bias=True))
    X = X @ np.linalg.inv(cs).T
    return X @ np.linalg.cholesky(cov).T


def test_e2_lmg_public_entry_matches_analytic_oracle():
    """v1.6 P1-4: via contribution_summary (public), not _lmg_shares.
    Noiseless leg: y == sum(c) exactly -> shares == oracle, sum == 1."""
    r, betas = 0.6, (2.0, 1.0, 1.0)
    cov_x = np.array([[1.0, r, 0.0], [r, 1.0, 0.0], [0.0, 0.0, 1.0]])
    X = _exact_moment_columns(600, cov_x, np.random.default_rng(7))
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    for v, b in zip(("x1", "x2", "x3"), betas):
        df[f"y_slope_{v}"] = b
    df["y_intercept"] = 0.0
    df["y"] = sum(b * df[v] for v, b in zip(("x1", "x2", "x3"), betas))
    adf = AliasDataFrame(df)
    meta = _meta(nvars=("x1", "x2", "x3"))
    make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)
    tab, _ = contribution_summary(adf, meta, "y")
    oracle = _analytic_lmg(r)
    got = tab.set_index("term").loc[["x1", "x2", "x3"],
                                    "var_share_shapley"].to_numpy()
    np.testing.assert_allclose(got, oracle, rtol=0, atol=1e-9)
    np.testing.assert_allclose(got.sum(), 1.0, atol=1e-9)
    naive = tab.set_index("term").loc[["x1", "x2", "x3"],
                                      "var_share_naive"].to_numpy()
    assert not np.allclose(got, naive)
    cov_cc, cov_cy, var_y = _analytic_3term_cov(r, betas)
    assert not np.allclose(got, _partial_sum_variant(cov_cc, var_y))


def test_e2b_real_target_denominator_with_noise_D7():
    """D-7 discriminator: with orthogonal noise, var(y) > var(sum c) and the
    shares must sum to the EXPLAINED share R2 = var_c/var_y — the old
    (non-conformant) explained-part denominator would return sum == 1."""
    rng = np.random.default_rng(11)
    n = 800
    cov_x = np.eye(2)
    X = _exact_moment_columns(n, cov_x, rng)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    df["y_slope_x1"] = 1.0
    df["y_slope_x2"] = 1.0
    df["y_intercept"] = 0.0
    c = df.x1 + df.x2
    e = rng.normal(size=n)
    e = e - e.mean()
    # residualize e against contributions, rescale to exact var 1.0
    for col in (df.x1, df.x2):
        e = e - (e @ col) / (col @ col) * col
    e = e / np.sqrt((e @ e) / n)
    df["y"] = c + e
    var_c = float(((c - c.mean()) ** 2).mean())
    r2 = var_c / (var_c + 1.0)
    adf = AliasDataFrame(df)
    meta = _meta()
    make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)
    tab, _ = contribution_summary(adf, meta, "y")
    np.testing.assert_allclose(tab["var_share_shapley"].sum(), r2, atol=1e-9)
    assert tab["var_share_shapley"].sum() < 0.999  # old object would give 1


def test_e2c_multi_row_pergroup_subframe_end_to_end():
    """v1.6 P1-2: genuine multi-row per-group source subframe through
    contribution_summary (join on subframe index columns)."""
    rng = np.random.default_rng(23)
    n = 400
    g = np.repeat([0, 1], n // 2)
    x1 = rng.normal(size=n)
    df = pd.DataFrame({"g": g, "x1": x1})
    b1 = np.where(g == 0, 1.0, 3.0)
    df["y"] = b1 * df.x1
    adf = AliasDataFrame(df)
    coeffs = pd.DataFrame({"g": [0, 1], "y_slope_x1": [1.0, 3.0],
                           "y_intercept": [0.0, 0.0]})
    adf.register_subframe("GBC", AliasDataFrame(coeffs),
                          index_columns=["g"])
    meta = _meta(nvars=("x1",))
    make_contribution_aliases(adf, meta, "GBC", "y", corr_warn=None)
    tab, _ = contribution_summary(adf, meta, "y")
    # v1 object (per-group betas x GLOBAL mean): the group-dependent
    # baseline term beta_g*mu is legitimately unexplained by the centered
    # contribution — oracle computed INDEPENDENTLY from the raw arrays:
    mu = float(df.x1.mean())
    cvec = b1 * (df.x1.to_numpy() - mu)
    yvec = df.y.to_numpy()
    cc = cvec - cvec.mean(); yy = yvec - yvec.mean()
    expected = float((cc @ yy) ** 2 / ((cc @ cc) * (yy @ yy)))
    np.testing.assert_allclose(tab["var_share_shapley"].iloc[0], expected,
                               rtol=0, atol=1e-12)
    assert expected > 0.99  # sanity: near-total, not exactly 1


def _partial_sum_variant(cov_cc, var_y):
    """The banned shortcut, implemented only to prove it differs (E2/E11)."""
    import itertools, math
    p = len(cov_cc)
    val = {(): 0.0}
    for k in range(1, p + 1):
        for S in itertools.combinations(range(p), k):
            sub = np.ix_(S, S)
            val[S] = float(np.ones(k) @ cov_cc[sub] @ np.ones(k)) / var_y
    shares = np.zeros(p)
    fact = [math.factorial(i) for i in range(p + 1)]
    for j in range(p):
        for k in range(0, p):
            wgt = fact[k] * fact[p - k - 1] / fact[p]
            for S in itertools.combinations([i for i in range(p) if i != j], k):
                Sj = tuple(sorted(S + (j,)))
                shares[j] += wgt * (val[Sj] - val[S])
    return shares


# ---------------------------------------------------------------- E3
def test_e3_corr_warning_fires_at_creation():
    n = 500
    x1 = RNG.normal(size=n)
    x2 = 0.97 * x1 + 0.05 * RNG.normal(size=n)  # |r| > 0.9
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": x1 + x2,
                       "y_slope_x1": 1.0, "y_slope_x2": 1.0,
                       "y_intercept": 0.0})
    adf = AliasDataFrame(df)
    with pytest.warns(UserWarning, match="var_share_shapley"):
        make_contribution_aliases(adf, _meta(), "GB", "y", corr_warn=0.8)


# ---------------------------------------------------------------- E4
def test_e4_aliases_bind_fallback_prequalified_subframe():
    """Fallback (D-5) leg: single-row source subframe, pre-qualified
    formulas resolve through the subframe mechanism."""
    df = pd.DataFrame({"x1": RNG.normal(size=100),
                       "x2": RNG.normal(size=100)})
    df["gid"] = 0
    df["y"] = 2.0 * df.x1 - 1.0 * df.x2 + 3.0
    adf = AliasDataFrame(df)
    coeffs = pd.DataFrame({"gid": [0], "y_slope_x1": [2.0],
                           "y_slope_x2": [-1.0], "y_intercept": [3.0]})
    adf.register_subframe("GBC", AliasDataFrame(coeffs),
                          index_columns=["gid"])
    names = make_contribution_aliases(adf, _meta(), "GBC", "y",
                                      corr_warn=None)
    assert set(n.startswith("c_") or "baseline" in n for n in names) == {True}
    # summary via single-row broadcast resolution
    tab, corr = contribution_summary(adf, _meta(), "y")
    assert set(tab["term"]) == {"x1", "x2"}


@pytest.mark.skipif(not HAS_SOURCE_KW, reason="ADF predates 13.73 source=")
def test_e4b_aliases_bind_source_kw():
    """Primary path (13.73): registered subframe + UNQUALIFIED coefficient
    names resolved via add_alias(source=)."""
    df = pd.DataFrame({"x1": RNG.normal(size=100),
                       "x2": RNG.normal(size=100)})
    df["gid"] = 0
    df["y"] = 2.0 * df.x1 - 1.0 * df.x2 + 3.0
    adf = AliasDataFrame(df)
    coeffs = pd.DataFrame({"gid": [0], "y_slope_x1": [2.0],
                           "y_slope_x2": [-1.0], "y_intercept": [3.0]})
    adf.register_subframe("GBC", AliasDataFrame(coeffs),
                          index_columns=["gid"])
    names = make_contribution_aliases(adf, _meta(), "GBC", "y",
                                      corr_warn=None)
    assert len(names) == 3
    tab, _ = contribution_summary(adf, _meta(), "y")
    assert set(tab["term"]) == {"x1", "x2"}


# ---------------------------------------------------------------- E5
def test_e5_block_delta_nonlinear_toy():
    """v1-scope-only nonlinear toy (D-4 v1.6): sqrt(p0**2 + (p1*x)**p2),
    REAL assertion: emitted alias values == independent numpy ablation."""
    n = 300
    df = pd.DataFrame({"x": np.abs(RNG.normal(size=n)) + 0.1})
    p0, p1, p2 = 1.0, 2.0, 2.0
    formula = f"sqrt({p0}**2 + ({p1}*x)**{p2})"
    adf = AliasDataFrame(df)
    meta = {"formulas": {"pred": formula}, "parameters": {}}
    names = make_block_delta_aliases(adf, meta, "GB", "pred",
                                     blocks={"xblk": ["x"]})
    assert names == ["block_delta_pred_xblk"]
    x = df.x.to_numpy(np.float64)
    x0 = float(x.mean())
    expected = (np.sqrt(p0**2 + (p1*x)**p2)
                - np.sqrt(p0**2 + (p1*x0)**p2))
    if hasattr(adf, "materialize_alias"):
        adf.materialize_alias("block_delta_pred_xblk")
    got = np.asarray(adf.df["block_delta_pred_xblk"], dtype=np.float64)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    # D-4: overlapping blocks -> ValueError
    with pytest.raises(ValueError, match="disjoint"):
        make_block_delta_aliases(adf, meta, "GB", "pred",
                                 blocks={"a": ["x"], "b": ["x"]})
    # collision guard (P1-5)
    with pytest.raises(ValueError, match="collides"):
        make_block_delta_aliases(adf, meta, "GB", "pred",
                                 blocks={"xblk": ["x"]})


# ---------------------------------------------------------------- E6
def test_e6_two_architect_tables_selection_filters_never_recenters():
    df = _pergroup_frame()
    adf = AliasDataFrame(df)
    make_contribution_aliases(adf, _meta(), "GB", "y", corr_warn=None)
    t_full, _ = contribution_summary(adf, _meta(), "y")
    sel = adf.df["g"] == 0
    t_sel, _ = contribution_summary(adf, _meta(), "y", selection=sel)
    assert not np.allclose(t_full["std"], t_sel["std"])  # differ correctly
    # never re-centers: registry mu unchanged by the selection call
    mu_before = dict(adf.df.attrs["gb_explain"]["y"]["mu"])
    contribution_summary(adf, _meta(), "y", selection=sel)
    assert adf.df.attrs["gb_explain"]["y"]["mu"] == mu_before


# ---------------------------------------------------------------- E7
def test_e7_metadata_failures_loud():
    df = _pergroup_frame()
    adf = AliasDataFrame(df)
    with pytest.raises(ValueError, match="coefficients"):
        make_contribution_aliases(adf, {"columns": {}}, "GB", "y")
    bad = _meta()
    bad["columns"]["coefficients"]["y"].append("y_slope_x1")  # duplicate
    with pytest.raises(ValueError, match="duplicate"):
        make_contribution_aliases(adf, bad, "GB", "y")
    bad2 = _meta()
    bad2["columns"]["coefficients"]["y"][0] = "totally_unrelated"
    with pytest.raises(ValueError, match="unresolvable"):
        make_contribution_aliases(adf, bad2, "GB", "y")


def test_e7b_unresolvable_response_y_D7_and_mismatched_meta():
    df = _pergroup_frame()
    adf = AliasDataFrame(df)
    meta = _meta()
    make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)
    # D-7: response column absent -> loud ValueError at summary
    # (REAL execution — the previous dead-code guard is the P1-4 fix)
    df2 = df.drop(columns=["y"])
    adf2 = AliasDataFrame(df2)
    make_contribution_aliases(adf2, meta, "GB", "y", corr_warn=None)
    with pytest.raises(ValueError, match="response column"):
        contribution_summary(adf2, meta, "y")
    sel = adf.df["x1"] > 1e9  # empty selection also loud
    with pytest.raises(ValueError, match="removes all rows"):
        contribution_summary(adf, meta, "y", selection=sel)
    # P1-1: mismatched metadata (extra term) -> loud ValueError
    meta_bad = _meta(nvars=("x1", "x2", "x9"))
    with pytest.raises(ValueError, match="does not match"):
        contribution_summary(adf, meta_bad, "y")


def test_e7c_weights_validation_and_staleness_guard():
    # negative weight -> ValueError at creation (P2-4)
    df = _pergroup_frame()
    df["w"] = 1.0
    df.loc[df.index[0], "w"] = -1.0
    adf = AliasDataFrame(df)
    with pytest.raises(ValueError, match="finite and >= 0"):
        make_contribution_aliases(adf, _meta(weights="w"), "GB", "y",
                                  corr_warn=None)
    # staleness (P2-3): re-register source subframe after creation
    df2 = pd.DataFrame({"gid": [0] * 50, "x1": RNG.normal(size=50)})
    df2["y"] = 2.0 * df2.x1
    adf2 = AliasDataFrame(df2)
    c1 = pd.DataFrame({"gid": [0], "y_slope_x1": [2.0],
                       "y_intercept": [0.0]})
    adf2.register_subframe("GBC", AliasDataFrame(c1), index_columns=["gid"])
    meta1 = _meta(nvars=("x1",))
    make_contribution_aliases(adf2, meta1, "GBC", "y", corr_warn=None)
    c2 = pd.DataFrame({"gid": [0, 1], "y_slope_x1": [2.0, 9.0],
                       "y_intercept": [0.0, 0.0]})
    adf2.register_subframe("GBC", AliasDataFrame(c2), index_columns=["gid"])
    with pytest.raises(ValueError, match="stale"):
        contribution_summary(adf2, meta1, "y")


# ---------------------------------------------------------------- E8
def test_e8_zero_variance_term():
    df = _pergroup_frame()
    df["x3"] = 7.0  # constant
    df["y_slope_x3"] = 0.5
    adf = AliasDataFrame(df)
    meta = _meta(nvars=("x1", "x2", "x3"))
    make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)
    tab, corr = contribution_summary(adf, meta, "y")
    row = tab[tab.term == "x3"].iloc[0]
    assert row["zero_variance"] and row["var_share_shapley"] == 0.0
    assert np.isfinite(tab["var_share_shapley"]).all()
    # v1.6 P2-1 matrix contract, asserted in FULL (P1-3 fix):
    assert corr.shape == (3, 3)
    assert list(corr.index) == ["x1", "x2", "x3"]      # labels + order
    assert list(corr.columns) == ["x1", "x2", "x3"]
    assert (corr.loc["x3", ["x1", "x2"]] == 0.0).all() # zeroed row
    assert (corr.loc[["x1", "x2"], "x3"] == 0.0).all() # zeroed column
    assert corr.loc["x3", "x3"] == 0.0                 # v1.6: diagonal 0
    assert np.isfinite(corr.values).all()


# ---------------------------------------------------------------- E9
def test_e9_weighted_fit_semantics():
    """WLS toy with pre-computed analytic constants.
    Rows: x=(0,1,2) each with weights (1,1,2): weighted mean of x
    = (0+1+2+2)/4 = 1.25 (vs unweighted 1.0) — analytic constant."""
    df = pd.DataFrame({"x1": [0.0, 1.0, 2.0], "w": [1.0, 1.0, 2.0]})
    df["y"] = 3.0 * df.x1
    df["y_slope_x1"] = 3.0
    df["y_intercept"] = 0.0
    adf = AliasDataFrame(df)
    meta = _meta(nvars=("x1",), weights="w")
    make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)
    assert abs(adf.df.attrs["gb_explain"]["y"]["mu"]["x1"] - 1.25) < 1e-12
    t_w, _ = contribution_summary(adf, meta, "y")           # auto: weighted
    # analytic weighted var of x: E_w[x^2]-1.25^2 = (0+1+8)/4 - 1.5625 = 0.6875
    np.testing.assert_allclose(t_w["std"].iloc[0], 3.0 * np.sqrt(0.6875),
                               rtol=0, atol=1e-12)
    with pytest.warns(UserWarning, match="DIFFERENT objective"):
        t_u, _ = contribution_summary(adf, meta, "y", weighted=False)
    # analytic unweighted var of x = 2/3
    np.testing.assert_allclose(t_u["std"].iloc[0], 3.0 * np.sqrt(2.0 / 3.0),
                               rtol=0, atol=1e-12)
    # weighted=True on unweighted fit -> ValueError
    df2 = _pergroup_frame()
    adf2 = AliasDataFrame(df2)
    make_contribution_aliases(adf2, _meta(), "GB", "y", corr_warn=None)
    with pytest.raises(ValueError, match="weights_column"):
        contribution_summary(adf2, _meta(), "y", weighted=True)


# ---------------------------------------------------------------- E10
def test_e10_p_guard_fires_at_summary_call_time():
    nvars = tuple(f"v{i}" for i in range(16))
    df = pd.DataFrame({v: RNG.normal(size=30) for v in nvars})
    df["y"] = 0.0
    for v in nvars:
        df[f"y_slope_{v}"] = 1.0
        df["y"] += df[v]
    df["y_intercept"] = 0.0
    adf = AliasDataFrame(df)
    meta = _meta(nvars=nvars)
    make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)  # no raise here
    with pytest.raises(ValueError, match="15"):
        contribution_summary(adf, meta, "y")


# ---------------------------------------------------------------- E11
def test_e11_perfect_collinearity_finite_deterministic():
    n = 400
    x1 = RNG.normal(size=n)
    df = pd.DataFrame({"x1": x1, "x2": 2.0 * x1})   # exactly collinear
    df["y"] = 1.0 * df.x1 + 0.5 * df.x2
    df["y_slope_x1"] = 1.0
    df["y_slope_x2"] = 0.5
    df["y_intercept"] = 0.0
    adf = AliasDataFrame(df)
    meta = _meta()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # corr warning expected, not under test
        make_contribution_aliases(adf, meta, "GB", "y")
    t1, _ = contribution_summary(adf, meta, "y")
    t2, _ = contribution_summary(adf, meta, "y")
    assert np.isfinite(t1["var_share_shapley"]).all()
    assert bool(t1["rank_deficient"].iloc[0]) is True
    np.testing.assert_allclose(t1["var_share_shapley"].sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(t1["var_share_shapley"],
                               t2["var_share_shapley"], rtol=0, atol=0)
    # oracle re-derived from the pinned formula at r=1 via _analytic path:
    # both contribution columns are proportional -> shares split 50/50 by
    # symmetry of equal contribution stds (c1 = x1, c2 = x1): hand value 0.5
    np.testing.assert_allclose(sorted(t1["var_share_shapley"]), [0.5, 0.5],
                               atol=1e-9)


# ------------------------------------------------- v1.6 round-3 fixes
def test_f1_meta_full_contract_mismatches_raise():
    df = _pergroup_frame(suffix="_v4")
    adf = AliasDataFrame(df)
    meta = _meta(suffix="_v4")
    make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)
    # suffix mismatch
    with pytest.raises(ValueError, match="suffix"):
        contribution_summary(adf, _meta(suffix=""), "y")
    # weights-column mismatch
    m2 = _meta(suffix="_v4"); m2["parameters"]["weights_column"] = "w"
    with pytest.raises(ValueError, match="weights_column"):
        contribution_summary(adf, m2, "y")
    # intercept mismatch
    m3 = _meta(suffix="_v4", fit_intercept=False)
    with pytest.raises(ValueError, match="intercept"):
        contribution_summary(adf, m3, "y")


def test_f2_all_zero_weights_raise_at_creation():
    df = _pergroup_frame()
    df["w"] = 0.0
    adf = AliasDataFrame(df)
    with pytest.raises(ValueError, match="zero total weight"):
        make_contribution_aliases(adf, _meta(weights="w"), "GB", "y",
                                  corr_warn=None)


def test_f3_string_selection_supported():
    df = _pergroup_frame()
    adf = AliasDataFrame(df)
    meta = _meta()
    make_contribution_aliases(adf, meta, "GB", "y", corr_warn=None)
    t_str, _ = contribution_summary(adf, meta, "y", selection="(g == 0)")
    t_bool, _ = contribution_summary(adf, meta, "y",
                                     selection=(adf.df["g"] == 0))
    pd.testing.assert_frame_equal(t_str, t_bool, check_exact=True)


def test_f4_join_duplicate_and_unmatched_keys_raise():
    rng = np.random.default_rng(31)
    df = pd.DataFrame({"g": np.repeat([0, 1], 20),
                       "x1": rng.normal(size=40)})
    df["y"] = df.x1
    adf = AliasDataFrame(df)
    dup = pd.DataFrame({"g": [0, 0, 1], "y_slope_x1": [1.0, 2.0, 1.0],
                        "y_intercept": [0.0, 0.0, 0.0]})
    adf.register_subframe("GBC", AliasDataFrame(dup), index_columns=["g"])
    meta = _meta(nvars=("x1",))
    make_contribution_aliases(adf, meta, "GBC", "y", corr_warn=None)
    with pytest.raises(ValueError, match="duplicate join keys"):
        contribution_summary(adf, meta, "y")
    # unmatched keys: subframe missing group 1
    df2 = df.copy()
    adf2 = AliasDataFrame(df2)
    part = pd.DataFrame({"g": [0], "y_slope_x1": [1.0],
                         "y_intercept": [0.0]})
    adf2.register_subframe("GBC", AliasDataFrame(part), index_columns=["g"])
    make_contribution_aliases(adf2, meta, "GBC", "y", corr_warn=None)
    # single-row broadcast path would mask this; force multi-row: add row
    part2 = pd.DataFrame({"g": [0, 2], "y_slope_x1": [1.0, 5.0],
                          "y_intercept": [0.0, 0.0]})
    adf3 = AliasDataFrame(df.copy())
    adf3.register_subframe("GBC", AliasDataFrame(part2), index_columns=["g"])
    make_contribution_aliases(adf3, meta, "GBC", "y", corr_warn=None)
    with pytest.raises(ValueError, match="unmatched join"):
        contribution_summary(adf3, meta, "y")
    # P1-3 (round 3): unmatched must STILL raise when the source column
    # contains a genuine NaN coefficient elsewhere
    part3 = pd.DataFrame({"g": [0, 2], "y_slope_x1": [np.nan, 5.0],
                          "y_intercept": [0.0, 0.0]})
    adf4 = AliasDataFrame(df.copy())
    adf4.register_subframe("GBC", AliasDataFrame(part3), index_columns=["g"])
    make_contribution_aliases(adf4, meta, "GBC", "y", corr_warn=None)
    with pytest.raises(ValueError, match="unmatched join"):
        contribution_summary(adf4, meta, "y")


@pytest.mark.skipif(not HAS_SOURCE_KW, reason="needs 13.73 source=")
def test_f5_nonlinear_source_binding_out_of_frame_param():
    """P2-4: block-delta formula referencing an out-of-frame parameter
    resolved via source= binding."""
    rng = np.random.default_rng(41)
    df = pd.DataFrame({"gid": 0, "x": np.abs(rng.normal(size=80)) + 0.1})
    adf = AliasDataFrame(df)
    pars = pd.DataFrame({"gid": [0], "p0c": [1.5]})
    adf.register_subframe("PARS", AliasDataFrame(pars),
                          index_columns=["gid"])
    meta = {"formulas": {"pred": "sqrt(p0c**2 + (2.0*x)**2)"},
            "parameters": {}}
    names = make_block_delta_aliases(adf, meta, "PARS", "pred",
                                     blocks={"xb": ["x"]})
    if hasattr(adf, "materialize_alias"):
        adf.materialize_alias(names[0])
    x = df.x.to_numpy(np.float64); x0 = float(x.mean())
    expected = (np.sqrt(1.5**2 + (2.0*x)**2)
                - np.sqrt(1.5**2 + (2.0*x0)**2))
    np.testing.assert_allclose(np.asarray(adf.df[names[0]], np.float64),
                               expected, rtol=1e-5, atol=1e-5)


def test_f6_block_delta_summary_same_tables_P0_1():
    """P0-1: the D-2 'same summary from the deltas' contract, end-to-end."""
    rng = np.random.default_rng(53)
    n = 500
    df = pd.DataFrame({"x": np.abs(rng.normal(size=n)) + 0.1,
                       "z": rng.normal(size=n)})
    formula = "sqrt(1.0 + (2.0*x)**2) + 0.5*z"
    df["pred"] = np.sqrt(1.0 + (2.0*df.x)**2) + 0.5*df.z
    adf = AliasDataFrame(df)
    meta = {"formulas": {"pred": formula}, "parameters": {}}
    make_block_delta_aliases(adf, meta, "GB", "pred",
                             blocks={"xb": ["x"], "zb": ["z"]})
    tab, corr = contribution_summary(adf, meta, "pred")
    assert list(tab["term"]) == ["xb", "zb"]
    assert corr.shape == (2, 2)
    assert np.isfinite(tab["var_share_shapley"]).all()
    # noiseless decomposition: shares sum to the explained share (<= 1)
    s = float(tab["var_share_shapley"].sum())
    assert 0.9 < s <= 1.0 + 1e-9
    # meta mismatch on the delta path is loud (P1-1 delta check)
    bad = {"formulas": {"pred": "x"}, "parameters": {}}
    with pytest.raises(ValueError, match="does not match"):
        contribution_summary(adf, bad, "pred")


def test_f7_common_mask_at_creation_P1_1():
    """Creation-time mu must use the COMMON mask: a NaN in x2 or y must
    exclude that row from x1's mean too."""
    df = pd.DataFrame({"x1": [0.0, 10.0, 20.0], "x2": [1.0, np.nan, 1.0],
                       "y": [0.0, 0.0, 0.0]})
    df["y_slope_x1"] = 1.0
    df["y_slope_x2"] = 1.0
    df["y_intercept"] = 0.0
    adf = AliasDataFrame(df)
    make_contribution_aliases(adf, _meta(), "GB", "y", corr_warn=None)
    mu = adf.df.attrs["gb_explain"]["y"]["mu"]
    assert mu["x1"] == 10.0  # rows 0,2 only (common mask), NOT 10.0==mean all3
    # per-term masking would ALSO give 10.0 here; discriminate via y-NaN:
    df2 = df.copy(); df2.loc[2, "y"] = np.nan; df2.loc[1, "x2"] = 1.0
    adf2 = AliasDataFrame(df2)
    make_contribution_aliases(adf2, _meta(), "GB", "y", corr_warn=None)
    mu2 = adf2.df.attrs["gb_explain"]["y"]["mu"]
    assert mu2["x1"] == 5.0  # rows 0,1 — y-NaN row excluded by common mask
