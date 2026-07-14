"""gb_explain — contribution aliases and variance decomposition for GB fits.

PHASE_13_74_GB_Explain, proposal v1.5 (approved 2026-07-05). Owner: GB.
Consumes ADF via public API only (add_alias / register_subframe); no dfdraw
or ADF feature changes (proposal §6).

Statistical objects (proposal §2/§3a — read before modifying):
- Per-row contribution aliases c_j = beta_j * (x_j - mu_j) are the
  INDEPENDENT-CONVENTION linear contribution (Linear SHAP under the
  interventional convention, Lundberg & Lee 2017). For correlated
  predictors this is NOT a correlation-aware attribution; the fair
  ranking is the `var_share_shapley` table column (exact LMG).
- `var_share_shapley` (LMG / Shapley-value regression): for every subset S,
  R2(S) = cov(X_S, y)^T . pinv(cov(X_S, X_S)) . cov(X_S, y) / var(y),
  computed refit-equivalently from the stored (weighted) covariances;
  share_j = Shapley average of R2(S+{j}) - R2(S) over all orderings.
  It is NOT Var(sum of fixed-coefficient contributions)/Var(y) — the
  partial-sum shortcut is NON-CONFORMANT (v1.5 negative clause); the two
  coincide only at r = 0.
- Nonlinear path = deterministic leave-one-block-out MEAN-REPLACEMENT
  ablation deltas ("block_delta"); the name SHAP appears nowhere here.

Naming: coefficient <-> variable mapping follows the frozen Output Column
Naming Contract, TECHNICAL_SUMMARY v4.1 ({t}_slope_{var}{s},
{t}_intercept{s}) — cited, not restated (drift lesson).
"""
from __future__ import annotations

import inspect
import itertools
import math
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_P_MAX = 15  # hard LMG guard (2^15 = 32768 subsets); proposal §3a
_PINV_RCOND = 1e-10  # symmetric pinv tolerance, documented; proposal §3a v1.5
_REGISTRY_KEY = "gb_explain"


# ---------------------------------------------------------------------------
# metadata parsing (loud-failure contract, §3a)
# ---------------------------------------------------------------------------
def _parse_meta(meta: dict, target: str) -> dict:
    """Extract the §3a required-shape fields. Loud ValueError otherwise."""
    try:
        coef_cols = list(meta["columns"]["coefficients"][target])
    except (KeyError, TypeError):
        raise ValueError(
            f"gb_explain: meta['columns']['coefficients'][{target!r}] is "
            f"required (proposal §3a metadata shape) and was not found.")
    suffix = meta.get("parameters", {}).get("suffix", "")
    weights_column = meta.get("parameters", {}).get("weights_column", None)

    slope_prefix = f"{target}_slope_"
    intercept_name = f"{target}_intercept{suffix}"
    terms: Dict[str, str] = {}          # variable -> slope column
    intercept_col: Optional[str] = None
    for col in coef_cols:
        if col == intercept_name:
            intercept_col = col
            continue
        if col.startswith(slope_prefix):
            var = col[len(slope_prefix):]
            if suffix:
                if not var.endswith(suffix):
                    raise ValueError(
                        f"gb_explain: coefficient column {col!r} does not "
                        f"carry the metadata suffix {suffix!r} (frozen "
                        f"naming contract, TS v4.1).")
                var = var[: -len(suffix)]
            if not var:
                raise ValueError(
                    f"gb_explain: cannot resolve a variable name from "
                    f"coefficient column {col!r}.")
            if var in terms:
                raise ValueError(
                    f"gb_explain: duplicate term {var!r} resolved from "
                    f"coefficient columns (have {terms[var]!r} and {col!r}).")
            terms[var] = col
        else:
            raise ValueError(
                f"gb_explain: unresolvable coefficient column {col!r} — "
                f"expected '{target}_slope_<var>{suffix}' or "
                f"'{intercept_name}' per the frozen naming contract "
                f"(TS v4.1).")
    if not terms:
        raise ValueError(
            f"gb_explain: no slope terms resolved for target {target!r}.")
    return {"terms": terms, "intercept_col": intercept_col, "suffix": suffix,
            "weights_column": weights_column}


def _weighted_mean(x: np.ndarray, w: Optional[np.ndarray]) -> float:
    x = np.asarray(x, dtype=np.float64)
    if w is None:
        return float(np.nanmean(x))
    w = np.asarray(w, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(w)
    return float(np.sum(w[m] * x[m]) / np.sum(w[m]))


def _weighted_cov(mat: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    """Weighted covariance (float64) of columns of `mat` (n x k)."""
    mat = np.asarray(mat, dtype=np.float64)
    if w is None:
        w = np.ones(mat.shape[0], dtype=np.float64)
    else:
        w = np.asarray(w, dtype=np.float64)
    m = np.isfinite(mat).all(axis=1) & np.isfinite(w)
    mat, w = mat[m], w[m]
    wsum = np.sum(w)
    mu = np.sum(mat * w[:, None], axis=0) / wsum
    d = mat - mu
    return (d * w[:, None]).T @ d / wsum


# ---------------------------------------------------------------------------
# LMG — exact, pinned object (§3a v1.5)
# ---------------------------------------------------------------------------
def _lmg_shares(cov_xx: np.ndarray, cov_xy: np.ndarray, var_y: float
                ) -> Tuple[np.ndarray, bool]:
    """Exact LMG shares from covariances.

    R2(S) = cov(X_S,y)^T . pinv(cov(X_S,X_S), rcond=1e-10) . cov(X_S,y)/var_y
    share_j = sum over S not containing j of
              |S|! (p-|S|-1)! / p!  *  (R2(S+{j}) - R2(S))
    NOT the fixed-coefficient partial-sum variance (non-conformant, §3a).
    Returns (shares, rank_deficient_flag).
    """
    p = len(cov_xy)
    if p > _P_MAX:
        raise ValueError(
            f"gb_explain: {p} terms exceeds the LMG hard guard p <= "
            f"{_P_MAX} (2^{_P_MAX} subsets; proposal §3a).")
    rank_deficient = bool(
        np.linalg.matrix_rank(cov_xx, tol=_PINV_RCOND * float(
            np.max(np.abs(np.diag(cov_xx))) or 1.0)) < p)

    r2 = {(): 0.0}
    idx = list(range(p))
    for size in range(1, p + 1):
        for S in itertools.combinations(idx, size):
            sub = np.ix_(S, S)
            c_xy = cov_xy[list(S)]
            pinv = np.linalg.pinv(cov_xx[sub], rcond=_PINV_RCOND,
                                  hermitian=True)
            r2[S] = float(c_xy @ pinv @ c_xy) / var_y if var_y > 0 else 0.0

    shares = np.zeros(p, dtype=np.float64)
    fact = [math.factorial(k) for k in range(p + 1)]
    for j in idx:
        others = [i for i in idx if i != j]
        for size in range(0, p):
            wgt = fact[size] * fact[p - size - 1] / fact[p]
            for S in itertools.combinations(others, size):
                Sj = tuple(sorted(S + (j,)))
                shares[j] += wgt * (r2[Sj] - r2[S])
    return shares, rank_deficient


# ---------------------------------------------------------------------------
# public API (§3)
# ---------------------------------------------------------------------------
def make_contribution_aliases(adf, meta, source, target, *, centered=True,
                              prefix="c_", dtype="float32",
                              corr_warn=0.8) -> List[str]:
    """Create per-term contribution aliases c_j = slope_j*(x_j - mu_j).

    Independent-convention linear contribution (NOT correlation-aware
    SHAP — see module docstring). Adds a baseline alias
    `{prefix}baseline_{target}` with baseline = intercept + sum(slope_j*mu_j)
    (no intercept term for fit_intercept=False fits, §3a) so that
    baseline + sum(c_j) == prediction (test E1).

    mu_j: computed ONCE on the full frame at creation (weighted mean when
    meta parameters.weights_column is set) and inlined as constants;
    `selection=` downstream never re-centers (§3a).
    """
    parsed = _parse_meta(meta, target)
    terms, suffix = parsed["terms"], parsed["suffix"]
    wcol = parsed["weights_column"]
    df = adf.df
    w = df[wcol].to_numpy() if wcol else None

    missing = [v for v in terms if v not in df.columns]
    if missing:
        raise ValueError(
            f"gb_explain: term variable(s) {missing} not present in the "
            f"data frame.")

    mu: Dict[str, float] = {
        v: (_weighted_mean(df[v].to_numpy(), w) if centered else 0.0)
        for v in terms}

    # correlation warning at creation (§2 guard; bug-#9 loud precedent)
    if corr_warn:
        varnames = list(terms)
        cov = _weighted_cov(df[varnames].to_numpy(), w)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(sd, sd)
        hits = [(varnames[i], varnames[j], corr[i, j])
                for i in range(len(varnames)) for j in range(i + 1, len(varnames))
                if np.isfinite(corr[i, j]) and abs(corr[i, j]) > corr_warn]
        if hits:
            desc = ", ".join(f"({a},{b}: r={r:+.3f})" for a, b, r in hits)
            warnings.warn(
                f"gb_explain: term pair(s) exceed |r|>{corr_warn}: {desc}. "
                f"Per-row contribution aliases are the independent-"
                f"convention quantity and are NOT correlation-aware; use "
                f"the var_share_shapley table column for fair attribution "
                f"(proposal §2).", UserWarning, stacklevel=2)

    # alias emission: primary = add_alias(..., source=) [PHASE_13_73];
    # fallback = pre-qualified '{source}.{col}' formulas (D-5)
    has_source_kw = "source" in inspect.signature(adf.add_alias).parameters
    frame_cols = set(df.columns)

    def qual(col):
        if col in frame_cols:      # coefficients already materialized in-frame
            return col
        return col if has_source_kw else f"{source}.{col}"

    def alias_kw(coef_cols):
        """Pass source= ONLY when the alias actually references an
        out-of-frame coefficient (13.73 validates the subframe exists —
        alma2 bug 2026-07-13: unconditional source= exploded on in-frame
        layouts with no registered subframe)."""
        needs = has_source_kw and any(c not in frame_cols for c in coef_cols)
        return {"source": source} if needs else {}

    created, formulas = [], {}
    for v, slope_col in terms.items():
        name = f"{prefix}{v}"
        expr = f"{qual(slope_col)} * ({v} - {mu[v]!r})"
        adf.add_alias(name, expr, dtype=dtype, **alias_kw([slope_col]))
        created.append(name)
        formulas[v] = expr
    base_name = f"{prefix}baseline_{target}"
    base_terms = [f"{qual(slope)} * {mu[v]!r}" for v, slope in terms.items()]
    if parsed["intercept_col"] is not None:
        base_terms.insert(0, qual(parsed["intercept_col"]))
    base_expr = " + ".join(base_terms) if base_terms else "0.0"
    _base_cols = list(terms.values()) + (
        [parsed["intercept_col"]] if parsed["intercept_col"] else [])
    adf.add_alias(base_name, base_expr, dtype=dtype, **alias_kw(_base_cols))
    created.append(base_name)

    reg = df.attrs.setdefault(_REGISTRY_KEY, {})
    reg[target] = {"terms": dict(terms), "mu": mu, "prefix": prefix,
                   "weights_column": wcol, "suffix": suffix,
                   "formulas": formulas, "baseline": base_name,
                   "source": source}
    return created


def contribution_summary(adf, meta, target, *, selection=None, weighted=None
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """The two architect tables (D-2): sigma_table and corr_matrix.

    Requires make_contribution_aliases(target) to have been called on this
    adf (registry-based; loud ValueError otherwise — CRR disclosure D-i1).
    All statistics computed in float64 from re-evaluated contribution
    FORMULAS, not from the (possibly float32) stored aliases (§3a numerics).
    `selection=` filters rows only — it never re-centers (§3a).
    """
    reg = adf.df.attrs.get(_REGISTRY_KEY, {}).get(target)
    if reg is None:
        raise ValueError(
            f"gb_explain: no contribution aliases registered for target "
            f"{target!r} on this frame — call make_contribution_aliases "
            f"first (proposal §3 usage order).")
    wcol = reg["weights_column"]
    if weighted is None:
        use_w = wcol is not None
    elif weighted is True:
        if wcol is None:
            raise ValueError(
                "gb_explain: weighted=True but the fit has no "
                "meta parameters.weights_column (§3 contract).")
        use_w = True
    else:  # weighted is False
        if wcol is not None:
            warnings.warn(
                "gb_explain: weighted=False overrides a weighted fit — the "
                "summary now decomposes a DIFFERENT objective than the "
                "fit's own WLS objective (§3a).", UserWarning, stacklevel=2)
        use_w = False

    df = adf.df if selection is None else adf.df[selection]
    if len(df) == 0:
        raise ValueError("gb_explain: selection removes all rows.")

    varnames = list(reg["terms"])
    if len(varnames) > _P_MAX:  # E10: fires at call time, limit named
        raise ValueError(
            f"gb_explain: {len(varnames)} terms exceeds the LMG hard guard "
            f"p <= {_P_MAX} (proposal §3a).")

    # float64 re-evaluation of c_j and prediction target variance
    w = df[wcol].to_numpy(np.float64) if (use_w and wcol) else None
    cvals = np.empty((len(df), len(varnames)), dtype=np.float64)
    for k, v in enumerate(varnames):
        slope = _resolve_series(adf, df, reg, reg["terms"][v])
        x = df[v].to_numpy(np.float64)
        cvals[:, k] = slope * (x - reg["mu"][v])
    y = cvals.sum(axis=1)  # explained (centered) part; var decomposition
    cov_full = _weighted_cov(np.column_stack([cvals, y]), w)
    cov_cc, cov_cy, var_y = (cov_full[:-1, :-1], cov_full[:-1, -1],
                             float(cov_full[-1, -1]))

    sd = np.sqrt(np.clip(np.diag(cov_cc), 0, None))
    zero_var = sd <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array(cov_cc / np.outer(sd, sd))  # writable copy
    if zero_var.any():  # excluded from correlation denominator — no NaN
        corr[zero_var, :] = 0.0
        corr[:, zero_var] = 0.0
        np.fill_diagonal(corr, 1.0)
    corr_matrix = pd.DataFrame(corr, index=varnames, columns=varnames)

    live = ~zero_var
    shares = np.zeros(len(varnames))
    rank_def = False
    if live.any() and var_y > 0:
        sub = np.ix_(live.nonzero()[0], live.nonzero()[0])
        shares_live, rank_def = _lmg_shares(
            cov_cc[sub], cov_cy[live], var_y)
        shares[live] = shares_live

    naive = np.where(var_y > 0, np.diag(cov_cc) / var_y, 0.0)
    mcp = []
    for k in range(len(varnames)):
        row = np.abs(corr_matrix.values[k].copy()); row[k] = -1
        mcp.append(varnames[int(row.argmax())] if len(varnames) > 1 else "")
    sigma_table = pd.DataFrame({
        "term": varnames,
        "std": sd,
        "var_share_naive": naive,
        "var_share_shapley": shares,  # D-3 RULED KEEP
        "max_corr_partner": mcp,
        "zero_variance": zero_var,
        "rank_deficient": rank_def,
    })
    return sigma_table, corr_matrix


def make_block_delta_aliases(adf, meta, source, target, blocks: Dict[str, list]
                             ) -> List[str]:
    """Nonlinear path (D-4): leave-one-block-out MEAN-REPLACEMENT ablation.

    delta_B = pred - pred_without_B, where pred_without_B evaluates
    meta['formula'] with every variable in block B replaced by its
    (weighted) mean — deterministic ablation. These are ablation deltas;
    they are not Shapley values of any kind (terminology contract, B4).
    Requires meta['formula'] (prediction expression); loud ValueError
    otherwise.
    """
    formula = meta.get("formula")
    if not formula:
        raise ValueError(
            "gb_explain: nonlinear block deltas require meta['formula'] "
            "(prediction expression); not found (§3 D-4 contract).")
    wcol = meta.get("parameters", {}).get("weights_column", None)
    df = adf.df
    w = df[wcol].to_numpy() if wcol else None
    created = []
    all_vars = sorted({v for vs in blocks.values() for v in vs})
    missing = [v for v in all_vars if v not in df.columns]
    if missing:
        raise ValueError(f"gb_explain: block variable(s) {missing} not in frame.")
    mu = {v: _weighted_mean(df[v].to_numpy(), w) for v in all_vars}
    for bname, bvars in blocks.items():
        ablated = formula
        for v in sorted(bvars, key=len, reverse=True):  # longest first
            ablated = _replace_var(ablated, v, repr(mu[v]))
        name = f"block_delta_{bname}"
        adf.add_alias(name, f"({formula}) - ({ablated})", dtype="float32")
        created.append(name)
    return created


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_series(adf, df, reg, col) -> np.ndarray:
    """Fetch a coefficient column as float64: direct, subframe, or broadcast
    scalar (single-group / global fit)."""
    if col in df.columns:
        return df[col].to_numpy(np.float64)
    src = reg.get("source")
    sf = getattr(adf, "subframes", None) or getattr(adf, "_subframes", None)
    entry = None
    if sf is not None:             # SubframeRegistry (.get) or plain dict
        getter = getattr(sf, "get", None)
        if callable(getter):
            entry = getter(src)
        elif hasattr(sf, "__getitem__"):
            try:
                entry = sf[src]
            except (KeyError, TypeError):
                entry = None
    if entry is not None:
        sdf = getattr(entry, "adf", entry)
        sdf = getattr(sdf, "df", sdf)
        if col in getattr(sdf, "columns", []) and len(sdf) == 1:
            return np.full(len(df), float(sdf[col].iloc[0]))  # global fit
    raise ValueError(
        f"gb_explain: coefficient column {col!r} is not reachable in "
        f"float64 for the summary (must be a frame column — e.g. the "
        f"materialized subframe join — or a single-row source subframe; "
        f"source={src!r}). CRR disclosure D-i2.")


_ID_BOUND = r"(?<![A-Za-z0-9_.])({})(?![A-Za-z0-9_])"


def _replace_var(expr: str, var: str, repl: str) -> str:
    import re
    return re.sub(_ID_BOUND.format(re.escape(var)), repl, expr)
