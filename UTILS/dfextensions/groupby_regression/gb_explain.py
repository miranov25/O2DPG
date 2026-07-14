"""gb_explain — contribution aliases and variance decomposition for GB fits.

PHASE_13_74_GB_Explain, proposal v1.6 (2026-07-05; conformance pass 2026-07-14). Owner: GB.
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
  partial-sum shortcut is NON-CONFORMANT (v1.6 negative clause). Precise
  coincidence condition (in-session derivation 2026-07-13, sharper than
  the proposal's "r = 0"): the two coincide exactly when every correlated
  contribution pair has EQUAL variances (r = 0 is the special case);
  they diverge whenever correlation meets unequal contribution scales.
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


def _validate_weights(w: Optional[np.ndarray]) -> None:
    """v1.6 §3a: weights must be finite and >= 0 (zero allowed, retained)."""
    if w is None:
        return
    w = np.asarray(w, dtype=np.float64)
    if not np.isfinite(w).all() or (w < 0).any():
        raise ValueError(
            "gb_explain: weights must be finite and >= 0 "
            "(negative/NaN/inf found; §3a v1.6).")


def _weighted_mean(x: np.ndarray, w: Optional[np.ndarray]) -> float:
    x = np.asarray(x, dtype=np.float64)
    if w is None:
        return float(np.nanmean(x))
    w = np.asarray(w, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(w)
    wsum = float(np.sum(w[m]))
    if wsum <= 0.0:
        raise ValueError(
            "gb_explain: zero total weight over valid rows — weighted mean "
            "undefined (§3a v1.6; P1-2 guard, applies at alias creation).")
    return float(np.sum(w[m] * x[m]) / wsum)


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
def make_contribution_aliases(adf, meta, source, target, *,
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

    _validate_weights(w)  # v1.6 P1-4: finite, >=0; ValueError otherwise
    # v1.6 §3a / round-3 P1-1: ONE common complete-case mask over
    # {predictors ∪ y (if present) ∪ weights} at creation — the SAME
    # policy as contribution_summary, not per-term independent masks.
    _mask_cols = [df[v].to_numpy(np.float64) for v in terms]
    if target in df.columns:
        _mask_cols.append(df[target].to_numpy(np.float64))
    _cmask = np.isfinite(np.column_stack(_mask_cols)).all(axis=1)
    if w is not None:
        _cmask &= np.isfinite(np.asarray(w, dtype=np.float64))
    if not _cmask.any():
        raise ValueError("gb_explain: no valid rows under the common "
                         "validity mask at alias creation (§3a v1.6).")
    _w_masked = (np.asarray(w, dtype=np.float64)[_cmask]
                 if w is not None else None)
    mu: Dict[str, float] = {
        v: _weighted_mean(df[v].to_numpy(np.float64)[_cmask], _w_masked)
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
                   "intercept_col": parsed["intercept_col"],
                   "weights_column": wcol, "suffix": suffix,
                   "formulas": formulas, "baseline": base_name,
                   "source": source,
                   "source_fp": _fingerprint_source(adf, source)}
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
    if reg.get("kind") == "delta":   # P0-1: nonlinear path, same tables
        return _delta_summary(adf, meta, target, reg,
                              selection=selection, weighted=weighted)
    # P1-1 (v1.6): meta is the source of truth and MUST be consumed —
    # re-parse and cross-check against the creation-time registry.
    parsed = _parse_meta(meta, target)
    _mism = []
    if parsed["terms"] != reg["terms"]:
        _mism.append(f"terms {parsed['terms']} vs registered {reg['terms']}")
    if parsed["suffix"] != reg.get("suffix"):
        _mism.append(f"suffix {parsed['suffix']!r} vs {reg.get('suffix')!r}")
    if parsed["intercept_col"] != reg.get("intercept_col"):
        _mism.append(f"intercept {parsed['intercept_col']!r} vs "
                     f"{reg.get('intercept_col')!r}")
    if parsed["weights_column"] != reg.get("weights_column"):
        _mism.append(f"weights_column {parsed['weights_column']!r} vs "
                     f"{reg.get('weights_column')!r}")
    if _mism:
        raise ValueError(
            "gb_explain: metadata does not match the creation-time registry "
            "for target %r — stale or mismatched metadata (P1-1 full "
            "contract check): %s" % (target, "; ".join(_mism)))
    wcol = parsed["weights_column"]
    # P2-3: subframe staleness guard
    _check_source_staleness(adf, reg)
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

    if selection is None:
        df = adf.df
    else:
        if isinstance(selection, str):  # P2-5: production string selections
            _ev = getattr(adf, "eval", None)
            mask = _ev(selection) if callable(_ev) else adf.df.eval(selection)
            selection = np.asarray(mask, dtype=bool)
        df = adf.df[selection]
    if len(df) == 0:
        raise ValueError("gb_explain: selection removes all rows.")
    # D-7 (v1.6): response y = the ORIGINAL fitted response column `target`
    if target not in df.columns:
        raise ValueError(
            f"gb_explain: response column {target!r} (D-7) is not "
            f"resolvable in the frame — the fitted dependent variable "
            f"itself is required for var(y).")

    varnames = list(reg["terms"])
    if len(varnames) > _P_MAX:  # E10: fires at call time, limit named
        raise ValueError(
            f"gb_explain: {len(varnames)} terms exceeds the LMG hard guard "
            f"p <= {_P_MAX} (proposal §3a).")

    w_raw = df[wcol].to_numpy(np.float64) if (use_w and wcol) else None
    _validate_weights(w_raw)
    y_real = df[target].to_numpy(np.float64)
    cvals = np.empty((len(df), len(varnames)), dtype=np.float64)
    for k, v in enumerate(varnames):
        slope = _resolve_series(adf, df, reg, reg["terms"][v])
        x = df[v].to_numpy(np.float64)
        cvals[:, k] = slope * (x - reg["mu"][v])
    # v1.6 §3a: ONE common complete-case validity mask over
    # {contributions ∪ y ∪ weights}, applied identically to every subset.
    stack = np.column_stack([cvals, y_real])
    mask = np.isfinite(stack).all(axis=1)
    if w_raw is not None:
        mask &= np.isfinite(w_raw)
    if not mask.any():
        raise ValueError("gb_explain: no valid rows after the common "
                         "validity mask (§3a v1.6).")
    stack = stack[mask]
    w = w_raw[mask] if w_raw is not None else None
    if w is not None and float(np.sum(w)) <= 0.0:
        raise ValueError("gb_explain: zero total weight after masking "
                         "(§3a v1.6).")
    cov_full = _weighted_cov(stack, w)
    cov_cc, cov_cy, var_y = (cov_full[:-1, :-1], cov_full[:-1, -1],
                             float(cov_full[-1, -1]))

    sd = np.sqrt(np.clip(np.diag(cov_cc), 0, None))
    zero_var = sd <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array(cov_cc / np.outer(sd, sd))  # writable copy
    if zero_var.any():  # v1.6 §3a matrix contract: full square; a
        # zero-variance term's row, COLUMN, AND DIAGONAL are 0 (P0-2 fix)
        corr[zero_var, :] = 0.0
        corr[:, zero_var] = 0.0
        d = np.diag_indices_from(corr)
        corr[d] = np.where(zero_var, 0.0, 1.0)
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
    """Nonlinear path (D-4, v1.6): leave-one-block-out MEAN-SUBSTITUTION
    ablation. delta_B(x) = pred(x) - pred(x_tilde) where x_tilde replaces
    every variable in block B by its (weighted, §3a) mean; pred is the
    expression from meta["formulas"] (dict keyed by target, or a plain
    string), bound via source= for out-of-frame names. Blocks must be
    DISJOINT (overlap -> ValueError). Output alias
    f"block_delta_{target}_{block}" with a name-collision guard (P1-5).
    These are ablation deltas; not Shapley values of any kind (B4)."""
    formulas = meta.get("formulas")
    if isinstance(formulas, dict):
        formula = formulas.get(target)
    else:
        formula = formulas
    if not formula:
        raise ValueError(
            "gb_explain: nonlinear block deltas require meta['formulas'] "
            "(prediction expression, dict-by-target or string); not found "
            "(D-4 v1.6 contract).")
    # disjointness (D-4 v1.6)
    seen: Dict[str, str] = {}
    for bname, bvars in blocks.items():
        for v in bvars:
            if v in seen:
                raise ValueError(
                    f"gb_explain: blocks must be disjoint — variable {v!r} "
                    f"appears in blocks {seen[v]!r} and {bname!r} (D-4).")
            seen[v] = bname
    wcol = meta.get("parameters", {}).get("weights_column", None)
    df = adf.df
    w = df[wcol].to_numpy() if wcol else None
    _validate_weights(w)
    all_vars = sorted(seen)
    missing = [v for v in all_vars if v not in df.columns]
    if missing:
        raise ValueError(
            f"gb_explain: block variable(s) {missing} not in frame.")
    mu = {v: _weighted_mean(df[v].to_numpy(), w) for v in all_vars}
    frame_cols = set(df.columns)
    has_source_kw = "source" in inspect.signature(adf.add_alias).parameters
    import re as _re
    ext_names = set(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", formula)) - frame_cols
    kw = ({"source": source}
          if (has_source_kw and ext_names & _external_candidates(adf, source))
          else {})
    created, delta_exprs = [], {}
    for bname, bvars in blocks.items():
        ablated = formula
        for v in sorted(bvars, key=len, reverse=True):  # longest first
            ablated = _replace_var(ablated, v, repr(mu[v]))
        name = f"block_delta_{target}_{bname}"
        if name in df.columns or name in getattr(adf, "aliases", {}):
            raise ValueError(
                f"gb_explain: block-delta output name {name!r} collides "
                f"with an existing column/alias (P1-5 guard).")
        expr = f"({formula}) - ({ablated})"
        adf.add_alias(name, expr, dtype="float32", **kw)
        created.append(name)
        delta_exprs[bname] = expr
    # P0-1 (round-3): register so contribution_summary serves the SAME two
    # tables from the deltas (D-2 "same summary" contract).
    reg = df.attrs.setdefault(_REGISTRY_KEY, {})
    reg[target] = {"kind": "delta", "terms": dict(delta_exprs),
                   "formula": formula, "weights_column": wcol,
                   "source": source,
                   "source_fp": _fingerprint_source(adf, source)}
    return created


def _external_candidates(adf, src):
    entry = _subframe_entry(adf, src)
    if entry is None:
        return set()
    return set(entry[0].columns)


def _delta_summary(adf, meta, target, reg, *, selection, weighted):
    """P0-1: sigma/corr tables over block-delta terms (kind='delta').
    meta consumption: meta['formulas'] must resolve to the registered base
    formula; mismatch -> ValueError. Statistics identical to the linear
    path: float64, common mask, D-7 real-target denominator, exact LMG."""
    formulas = meta.get("formulas")
    mform = formulas.get(target) if isinstance(formulas, dict) else formulas
    if mform != reg.get("formula"):
        raise ValueError(
            f"gb_explain: meta['formulas'] for {target!r} does not match "
            f"the registered block-delta base formula (P1-1 delta check).")
    wcol = reg.get("weights_column")
    if weighted is True and wcol is None:
        raise ValueError("gb_explain: weighted=True but no weights_column.")
    use_w = (wcol is not None) if weighted is None else bool(weighted)
    if weighted is False and wcol is not None:
        warnings.warn("gb_explain: weighted=False overrides a weighted "
                      "fit (§3a).", UserWarning, stacklevel=3)
    _check_source_staleness(adf, reg)
    if selection is None:
        df = adf.df
    else:
        if isinstance(selection, str):
            _ev = getattr(adf, "eval", None)
            mask = _ev(selection) if callable(_ev) else adf.df.eval(selection)
            selection = np.asarray(mask, dtype=bool)
        df = adf.df[selection]
    if len(df) == 0:
        raise ValueError("gb_explain: selection removes all rows.")
    if target not in df.columns:
        raise ValueError(
            f"gb_explain: response column {target!r} (D-7) is not "
            f"resolvable in the frame.")
    blocks = list(reg["terms"])
    if len(blocks) > _P_MAX:
        raise ValueError(f"gb_explain: {len(blocks)} blocks exceeds the "
                         f"LMG hard guard p <= {_P_MAX}.")
    w_raw = df[wcol].to_numpy(np.float64) if (use_w and wcol) else None
    _validate_weights(w_raw)
    y_real = df[target].to_numpy(np.float64)
    cvals = np.column_stack([
        _eval_float64(adf, df, reg, reg["terms"][b]) for b in blocks])
    stack = np.column_stack([cvals, y_real])
    mask = np.isfinite(stack).all(axis=1)
    if w_raw is not None:
        mask &= np.isfinite(w_raw)
    if not mask.any():
        raise ValueError("gb_explain: no valid rows after the common "
                         "validity mask (§3a v1.6).")
    stack = stack[mask]
    w = w_raw[mask] if w_raw is not None else None
    if w is not None and float(np.sum(w)) <= 0.0:
        raise ValueError("gb_explain: zero total weight after masking.")
    cov_full = _weighted_cov(stack, w)
    cov_cc, cov_cy, var_y = (cov_full[:-1, :-1], cov_full[:-1, -1],
                             float(cov_full[-1, -1]))
    sd = np.sqrt(np.clip(np.diag(cov_cc), 0, None))
    zero_var = sd <= 0
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.array(cov_cc / np.outer(sd, sd))
    if zero_var.any():
        corr[zero_var, :] = 0.0
        corr[:, zero_var] = 0.0
        d = np.diag_indices_from(corr)
        corr[d] = np.where(zero_var, 0.0, 1.0)
    corr_matrix = pd.DataFrame(corr, index=blocks, columns=blocks)
    live = ~zero_var
    shares = np.zeros(len(blocks))
    rank_def = False
    if live.any() and var_y > 0:
        sub = np.ix_(live.nonzero()[0], live.nonzero()[0])
        shares_live, rank_def = _lmg_shares(cov_cc[sub], cov_cy[live], var_y)
        shares[live] = shares_live
    naive = np.where(var_y > 0, np.diag(cov_cc) / var_y, 0.0)
    mcp = []
    for k in range(len(blocks)):
        r = np.abs(corr_matrix.values[k].copy()); r[k] = -1
        mcp.append(blocks[int(r.argmax())] if len(blocks) > 1 else "")
    sigma_table = pd.DataFrame({
        "term": blocks, "std": sd, "var_share_naive": naive,
        "var_share_shapley": shares, "max_corr_partner": mcp,
        "zero_variance": zero_var, "rank_deficient": rank_def})
    return sigma_table, corr_matrix


_NUMPY_ENV = {n: getattr(np, n) for n in (
    "sqrt", "abs", "exp", "log", "log10", "sin", "cos", "tan", "arctan",
    "arctan2", "arcsin", "arccos", "sinh", "cosh", "tanh", "clip",
    "minimum", "maximum", "sign", "floor", "ceil", "where", "pi", "e")}


def _eval_float64(adf, df, reg, expr) -> np.ndarray:
    """Evaluate a stored formula in float64: frame columns directly,
    out-of-frame names via the source subframe (broadcast or join)."""
    import re as _re
    names = set(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    env = dict(_NUMPY_ENV)
    for nm in names:
        if nm in env:
            continue
        if nm in df.columns:
            env[nm] = df[nm].to_numpy(np.float64)
        else:
            try:
                env[nm] = _resolve_series(adf, df, reg, nm)
            except ValueError:
                pass  # builtins/literals; NameError below if truly needed
    out = eval(expr, {"__builtins__": {}}, env)  # noqa: S307 — sandboxed env
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _subframe_entry(adf, src):
    """Return (frame_df, left_keys, right_keys) or None. Prefers the
    registry's get_entry dict ({'frame','index','right_index'} — 13.65
    asymmetric-key form); falls back to get()/getitem + attribute probing."""
    sf = getattr(adf, "subframes", None) or getattr(adf, "_subframes", None)
    if sf is None:
        return None
    ge = getattr(sf, "get_entry", None)
    if callable(ge):
        try:
            e = ge(src)
        except Exception:
            e = None
        if isinstance(e, dict) and "frame" in e:
            fr = e["frame"]
            fdf = getattr(fr, "df", fr)
            left = list(e.get("index") or [])
            right = list(e.get("right_index") or left)
            return (fdf, left or None, right or None)
    getter = getattr(sf, "get", None)
    obj = None
    if callable(getter):
        obj = getter(src)
    elif hasattr(sf, "__getitem__"):
        try:
            obj = sf[src]
        except (KeyError, TypeError):
            obj = None
    if obj is None:
        return None
    fdf = getattr(getattr(obj, "adf", obj), "df", getattr(obj, "adf", obj))
    keys = None
    for attr in ("index_columns", "join_columns", "left_on"):
        k = getattr(obj, attr, None)
        if isinstance(k, (list, tuple)) and k:
            keys = list(k)
            break
    return (fdf, keys, keys)


def _fingerprint_source(adf, src):
    entry = _subframe_entry(adf, src)
    if entry is None:
        return None
    sdf, left, right = entry
    try:  # content-sensitive (P2-3, GPT1): same-shape value swaps detected
        csum = int(pd.util.hash_pandas_object(sdf, index=False).sum())
    except Exception:
        csum = None
    return (len(sdf), tuple(sdf.columns), csum,
            tuple(left or ()), tuple(right or ()))


def _check_source_staleness(adf, reg):
    """v1.6 P2-3: fail loudly if the source subframe was re-registered or
    reshaped after alias creation."""
    fp_now = _fingerprint_source(adf, reg.get("source"))
    fp_then = reg.get("source_fp")
    if fp_then is not None and fp_now != fp_then:
        raise ValueError(
            f"gb_explain: source subframe {reg.get('source')!r} changed "
            f"since alias creation (was {fp_then}, now {fp_now}) — stale "
            f"registry; re-run make_contribution_aliases (P2-3 guard).")


def _resolve_series(adf, df, reg, col) -> np.ndarray:
    """Fetch a coefficient column as float64: direct, subframe, or broadcast
    scalar (single-group / global fit)."""
    if col in df.columns:
        return df[col].to_numpy(np.float64)
    src = reg.get("source")
    entry = _subframe_entry(adf, src)
    if entry is not None:
        sdf, left, right = entry
        if col in getattr(sdf, "columns", []):
            if len(sdf) == 1:
                return np.full(len(df), float(sdf[col].iloc[0]))  # global
            if left and right and all(k in df.columns for k in left):
                # P1-2 (v1.6 §4): per-group join, asymmetric keys supported
                rsub = sdf[list(right) + [col]]
                if rsub.duplicated(list(right)).any():  # P2-2: loud
                    raise ValueError(
                        f"gb_explain: source subframe {src!r} has duplicate "
                        f"join keys {list(right)} — ambiguous per-group "
                        f"coefficients (P2-2 guard).")
                mg = df[list(left)].merge(
                    rsub, left_on=list(left), right_on=list(right),
                    how="left", indicator=True)
                n_unmatched = int((mg["_merge"] == "left_only").sum())
                if n_unmatched:  # P1-3: indicator-based, NaN-coefficient safe
                    raise ValueError(
                        f"gb_explain: {n_unmatched} row(s) have no matching "
                        f"group in source subframe {src!r} (unmatched join "
                        f"keys guard).")
                return mg[col].to_numpy(np.float64)
    raise ValueError(
        f"gb_explain: coefficient column {col!r} is not reachable in "
        f"float64 for the summary (must be a frame column — e.g. the "
        f"materialized subframe join — or a single-row source subframe; "
        f"source={src!r}). CRR disclosure D-i2.")


_ID_BOUND = r"(?<![A-Za-z0-9_.])({})(?![A-Za-z0-9_])"


def _replace_var(expr: str, var: str, repl: str) -> str:
    import re
    return re.sub(_ID_BOUND.format(re.escape(var)), repl, expr)
