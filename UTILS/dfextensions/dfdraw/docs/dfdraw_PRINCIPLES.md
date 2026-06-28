# dfdraw_PRINCIPLES.md — Design Principles & Shortcut Grammar

**Version:** 1.2-draft rev c (1.1 + §2A Grammar of Graphics: descriptive [CURRENT] specification; rev c = normalization curve-source mechanism corrected + expression-in-slot documented)
**Date:** 2026-06-11
**Status:** PROPOSAL — awaits dfdraw-team review and architect ratification (§9)
**Drafter:** [Fable1] [AliasDataFrame] [Coder] — consumer-side draft at architect request; dfdraw team owns this document after ratification. (The dfdraw coder's v1.0 is superseded; its §0 drift narrative and §1 principle names are retained.)
**Review package (3 documents):** [1] this document; [2] `dfdraw_CALLFORM_EXTRACTION.md` (grammar-aligned evidence census from the four production scripts; rerunnable via `extract_callforms.py` + `callforms.json`); [3] `dfdraw_REVIEW_PACKAGE_SUMMARY.md` (cover note: what is binding, what is asked, sequencing). Source scripts attached: `time_series_draw.py` (gallery v2.2), `time_series.py`, `time_series_TroubleShooting.py`, `makeSmoothMapsWithTPC.py`.

---

## §0 Why this document exists

Phases 13.42 (`fit`), 13.43 (`summary_fit`), 13.51 (`time_format`), and earlier (`quantiles`) each implemented shortcut parsing ad-hoc: shortcut first, dict form added later with whatever schema seemed natural to that phase's coder — no constraint that the dict be a strict superset of the shortcut, no shared key vocabulary, no clash protocol. Result (dfdraw coder's own probe, 2026-06-11): `fit={'type':…}` raises (`'fun'` expected); `time_format` and `quantiles` have no dict form; `summary_fit` needs `'kind'`. Phase 13.52 (overlay) is the one surface designed with explicit shortcut⇄full symmetry — it works, and it is the template.

The fix is NOT a rewrite. The working conventions, censused across 84 production calls, reduce to seven orthogonal grammar rules (§2). This document ratifies the grammar, derives the principles' enforcement from it, and fixes the vocabulary — so the Phase 13.57.DF audit checks declarations against the grammar instead of re-litigating each kwarg, and future phases conform by construction.

---

## §0a Lineage (precedents — panel-confirmed)

The philosophy is well-precedented; the governance form is the novel part. Citations: Alan Kay — *"Simple things should be simple, complex things should be possible"* (P-0); Larry Wall, *Programming Perl* (1991); Wickham, *A Layered Grammar of Graphics*, J. Comp. Graph. Stat. 19(1), 2010 — shortcut geoms over a full grammar (R-3/R-4); Buitinck et al., *API design for machine learning software (scikit-learn)*, arXiv:1309.0238 (2013) — uniform API, sensible defaults, consistency; Waskom, seaborn, JOSS 6(60), 2021 — `**kwargs` pass-through to matplotlib (P-5); NumPy broadcasting (R-1); ROOT `TTree::Draw` (Brun & Rademakers, ~1995) — the architect's stated model. Novel here: binding the philosophy to an executable conformance corpus with CURRENT/TARGET status, verbatim architect quotes as binding constraints, and audit-checkable declarations (closest analog: sklearn's `check_estimator`).

## §1 Architect decisions (quoted with spelling normalized at the architect's request, 2026-06-11 — deviation from GP-3 typo preservation is itself architect-ratified; these BIND this document)

> *"I need full dict as a verbose option and very reasonable shortcuts."*
> — M. Ivanov, 2026-06-11

> *"We made some chaotic decisions in the past, but we need shortcuts, and those shortcuts should follow a pattern, and name clashes should be avoided. I do not want to provide the full dictionary every time; we need powerful shortcuts."*
> — M. Ivanov, 2026-06-11

> *"First, I want to be sure that the old code can work after small modifications."*
> — M. Ivanov, 2026-06-11

> *"We need better current-code understanding before destroying what is working."*
> — M. Ivanov, 2026-06-11

> *"There are many things I consider acceptable in current code and very useful. selection is a scalar. selection_vector is a vector. They are applied as a scalar or as a vector. If both are specified, we use a logical operation. I do not want to remove very useful conventions."*
> — M. Ivanov, 2026-06-11

> *"Whoever prepares the audit must know what the current code is doing."*
> — M. Ivanov, 2026-06-11

> *"We have: string → list of strings – that is normal and should be listed."*
> — M. Ivanov, 2026-06-11

> *"Following the principle is important, but making usage easy is even more important. In fact, this should be the basic principle. Default interactive usage should be simple."*
> — M. Ivanov, 2026-06-11 (ratifies P-0 as the ranking principle)

> *"I want to provide data sets to users and ask them to make a simple query – aligned with the code, using just a few switches. For more demanding use cases, we will need full dictionaries, but I also want tree->Draw-like simplicity."*
> — M. Ivanov, 2026-06-11

> *"Interactively I prefer not to consult an AI assistant on how to fill dictionaries for something where default values and simple shortcuts are sufficient."*
> — M. Ivanov, 2026-06-11

> *"The first question during approval is: did it make the user's life simpler? Do we benefit from standardization? I prefer standardization that benefits users. If not, it is better not to implement it."*
> — M. Ivanov, 2026-06-11 (the P-0 approval gate, verbatim)

> *"I prefer minimal changes to my already existing conventions."*
> — M. Ivanov, 2026-06-11 (binding review constraint, rev f / Phase 13.57; panel result consistent: 11/11, zero [BREAKING], zero un-preservable conventions)

> *"Can we create a logic or grammar for shortcuts that preserves functionality?"*
> — M. Ivanov, 2026-06-11 (answered by §2; ratification of §2 answers it YES)

> *"In the audit, we should prepare example use cases of the mapping. I do not want to redo that repeatedly. We have to fix the interface soon. The codified principles should prevent problems in the future."*
> — M. Ivanov, 2026-06-11

**Binding interpretation:** *"We must be able to express our needs"* (M. Ivanov, 2026-06-11) — the needs are the four production scripts plus the ADF call sites. §6 makes them an executable conformance set.

---

## §2 The Shortcut Grammar

**Form-status legend (rev c, panel-required):** every example is tagged
**[CURRENT]** (works at dfdraw HEAD — corpus or source citation given) or
**[TARGET]** (does NOT exist yet; a Phase 13.57 fix-phase **candidate** — every
[TARGET] form must pass the P-0 approval gate before implementation; a [TARGET]
form nobody needs is dropped, not built).
Auditors: [CURRENT] forms are conformance checks; [TARGET] forms are the
expected-state column — filing an [X] against a [TARGET] form for not
existing yet is a category error, and presenting a [TARGET] form as current
is a finding against THIS document. The §4 declarations carry an explicit
current/target column per form.

### G-0 — Preservation (the master invariant)

> **A value form that is valid today keeps its exact meaning forever. Grammar rules only ADD acceptable forms; they never remove or reinterpret one.**

This is the architect's "old code can work after small modifications" expressed as a property of the grammar rather than a promise in prose. Every rule below and every migration in §7 is subordinate to G-0.

### R-1 — Scalar → vector promotion (string → list of strings)

A kwarg declared *vectorizable* accepts a scalar or a list; a list means one curve / panel / layer / band-edge per element (the axis meaning is part of the kwarg's declaration).

```python
facet_by = 'side'  →  facet_by = ['side','qpt']  # [CURRENT] corpus: str ×28, list ×1 (ts_draw:126)
selection_vector = ['A','B']                      # [CURRENT] corpus ×8
quantiles = [0.05, 0.5, 0.95]                     # [CURRENT] the list IS the value (3/5/7-elem ×12)
expr = 'dy:t; dz:t'  ≡  expr = ['dy:t','dz:t']    # [TARGET] — the ';' spelling does NOT exist at
    # dfdraw HEAD (executed: ValueError "Invalid expression", fable5_5 rev-c probe). It is an
    # architect-required need per the §1 string→list quote ("that is normal and
    # should be listed"); 13.57 decides the entry form.
```

**Entry-form caveat (executed, fable5_5):** list-valued `expr` at the raw `DFDraw.draw` surface raises `AttributeError` today — the vector layer's entry forms differ per surface. Each vectorizable kwarg's §4 declaration must therefore state its entry form **per surface** (R-7), current and target separately.

Composition of several vector kwargs in one call follows the existing `vector_compose` semantics — [CURRENT]; the grammar names what is already implemented. **The default is `'inner'`** (element-wise pairing, equal lengths required; `drawer.py:1799`). The ADF surface auto-forces `'outer'` **only for single-Y (`n_y == 1`) + a multi-element `selection_vector`/`weights_vector`** (`AliasDataFrame.py:11409`/`:11461`, Phase 13.35), and always respects an explicit `vector_compose`. A multi-element **Y-vector therefore composes `inner` (paired)** — e.g. `y`-vector of length 4 × `selection_vector` of length 4 → **4 paired curves, not 16**.

### R-2 — Scalar + vector pair with a declared fold

Where both the scalar and the vector form of one concept exist, both may be given: the scalar applies globally and is folded into every vector element with the pair's declared operator.

| Pair | Fold | Status |
|---|---|---|
| `selection` ∧ `selection_vector[i]` | **logical AND** | **[CURRENT]** — source-verified: `_combine_selections(global_sel, per_curve_sel)` at `drawer.py` L1480, applied L1809 (fable5_5 executed probe); ratifies the architect's convention verbatim |
| `weights` × `weights_vector[i]` | multiplication | [CURRENT] behavior, ratified |
| any future pair without an obvious fold | `replace-error` (giving both raises) | default, §8 decision 2 |

**Scope (panel RC-3):** the `replace-error` default applies ONLY to future scalar+vector pairs not yet declared; the ratified AND-fold (`selection`∧`selection_vector`) and multiply-fold (`weights`×`weights_vector`) are [CURRENT] and unchanged forever.

### R-3 — Shortcut ≡ dict with one primary key (superset rule)

Every scalar shortcut is exactly sugar for a one-key dict; the dict may carry more keys (the architect's "full dict as a verbose option"):

All shortcut spellings below are **[CURRENT]** (corpus-verbatim). Current dict
keys at HEAD, stated explicitly (panel R-D): `fit` dict accepts `'fun'` +
`'initial'`/`'guess'` (fits.py:42 — the latter two rename scipy's `p0`, a P-5
violation resolved by aliasing, §8 D3); `summary_fit` dict accepts `'kind'`;
`time_format`/`quantiles` dicts raise. Canonical dict spellings below are
**[TARGET]** candidates; old keys become kept-forever aliases (G-0):

```python
fit = 'gauss'         ≡  fit = {'type': 'gauss'}             # [TARGET] dict; + {'p0': …, 'bounds': …}
                                                              # [CURRENT] dict: {'fun': …, 'initial': …}
time_format = '%H:%M' ≡  time_format = {'format': '%H:%M'}   # [TARGET] dict; + {'tz': …, 'locale': …}
summary_fit = 'table' ≡  summary_fit = {'type': 'table'}     # [TARGET] dict (current key: 'kind')
quantiles = [q…]      ≡  quantiles = {'levels': [q…]}        # [TARGET] dict; + {'mode': …}
central = 'median'                                            # scalar stays the ONLY form — §8 D5
    # (P-0 revision: zero corpus demand for a central dict; the D5 fix is the ADDITIVE
    #  y_central return_data column, scalar scope; fit seeding already correct since 13.51)
```

Testable invariance, generated mechanically from declarations: `draw(k=v)` produces bit-identical output to `draw(k={primary: v})`. These are the audit's per-pair equivalence tests (12–15 tests, one per surface).

### R-4 — Sibling-kwarg families ≡ flattened dict (the namespace / clash rule)

The dominant production pattern — `base=…, base_bins=…, base_quantiles=…` — is the **flattened spelling of the dict form**, with the prefix as the namespace:

Sibling spellings **[CURRENT]** (corpus, dominant pattern); dict spellings
**[TARGET]** (executed: `group_by={…}` raises TypeError at HEAD — fable5_5):

```python
group_by='abs(tgl)', group_by_bins=5   ≡  group_by={'by': 'abs(tgl)', 'bins': 5}   # [TARGET] dict
facet_by='qpt', facet_by_quantiles=9   ≡  facet_by={'by': 'qpt', 'quantiles': 9}   # [TARGET] dict
```

**Clash protocol (deterministic — no silent precedence):**
1. Dict form + sibling kwarg setting the **same key of the same family** in one call → hard error.
2. Different keys of the same family may mix freely (dict for some, siblings for others).
3. Two different shortcuts whose expansions touch the same underlying key → hard error at expansion time, naming both sources.
4. Companion kwargs lacking the prefix today (`quantile_mode` parametrizes `quantiles`) get a prefixed alias (`quantiles_mode`); the old name is kept forever per G-0.

**Worked clash example (panel R-K):**
```python
adf.draw('y:x', group_by={'by': 'tgl', 'bins': 5}, group_by_bins=9)
# → ValueError: group_by 'bins' given twice (dict and group_by_bins=) — rule 1, hard error
adf.draw('y:x', group_by={'by': 'tgl'}, group_by_bins=9)            # OK — rule 2, different keys mix
```

Expression values (`'abs(tgl)'` as `group_by`) are legal wherever a column name is — observed in production ×5, therefore contract **at the ADF surface**: the ADF alias machinery resolves expressions before delegation; the dfdraw layer itself requires a column name (Opus1 executed: raw `DFDraw.profile(group_by='abs(x)')` raises). Fix scope for any expression-related finding is ADF-only (panel layering clarification).

### R-5 — Type composition with `+`

`type='A+B'` composes layer types left-to-right (z-order). The only composition operator; symmetric with the full form `overlay(layers=[…])` [CURRENT]. Composition syntax lives nowhere else. Per-surface verification (panel R-H — claims are per-surface, never "all surfaces" without cites):

| Surface | Overlay string | Evidence |
|---|---|---|
| `adf.draw` | [CURRENT] | 13.55 T1; gallery fig37 |
| `adf.draw_figures` spec | [CURRENT] | 13.55 T5; gallery fig39 |
| `adf.draw_batch` spec | [CURRENT] | gallery fig36 (corpus ts_draw:258) |
| raw `DFDraw.draw` | [CURRENT] | Phase 13.52 origin |

### R-6 — Alias tolerance, ratified and append-only

One alias table per vocabulary position, shipped in this document (§5) and introspectable at runtime (`draw_help` — **[TARGET]** at the dfdraw layer: dfdraw has no `draw_help` at HEAD; the live-introspected `draw_help` exists at the ADF surface since Phase 13.56). Value-aliases `'gaus'→'gauss'`, `'histo'→'hist'` are **[CURRENT]** (production-live: ts:815, ts_draw:248); key-aliases (`'fun'→'type'`, `'kind'→'type'`) are **[TARGET]** (13.57 deliverable, old keys kept with warning). Aliases are **append-only** (G-0): removal is impossible by rule, so a one-character production spelling never breaks.

### R-7 — One vocabulary, three surfaces

The same kwarg names and the same grammar apply identically inside `draw(...)`, `draw_figures` plot specs, and `draw_batch` specs (corpus §6: specs already reuse `expr/type/bins/selection/range` verbatim; `'expr'` required in every spec). A rule ratified once applies everywhere; per-surface divergence is an audit finding by definition.

---

## §2A Grammar of Graphics — expressions, statistical transforms, and layout [CURRENT]

**Scope of the term (naming vote, 10 reviewers, 2026-06-21 — unanimous against "Grammar of Graphics and Statistics").** In this document "Grammar of Graphics" follows Wilkinson's original meaning (*The Grammar of Graphics*, Springer, 1999/2005), in which **statistics is a first-class component** (the seven components: variables, algebra, scales, **statistics**, geometry, coordinates, aesthetics) — not an add-on. dfdraw's statistical transforms (profile central values, quantile bands, fits, summary fits, normalization) are therefore *inside* the grammar, exactly as `stat_*` is inside ggplot2's layered grammar (Wickham 2010, §0a). The name is kept unqualified-but-subtitled; a deeper ggplot2/Wilkinson comparison is deferred to `dfdraw_COMPARISON.md`.

**Purpose (Pass 1).** §2 states the *meta-rules* (G-0, R-1..R-7): how forms generalize. §2A states the *concrete grammar those rules act on* — the as-built expression mini-language and channel vocabulary that until now lived only in `drawer.py` and docstrings. This section is **purely descriptive of dfdraw HEAD** (canonical `drawer.py` 8,082 lines / MD5 `e851acc72510d7a4c03690a6afb1e4b6`); every form is **[CURRENT]** with a source citation. It introduces **no** new or changed behavior — no `[TARGET]` forms appear here (those remain in §2/§8). Where the current grammar has a natural extension seam, it is noted only as a **statement of present behavior**; the design of extensions (y-vector layout; N-numerator normalization) is **Pass 2 (brainstorming) and out of scope here**. Subordinate to G-0 and P-0 like everything else.

### §2A.1 The plot expression (the positional mini-language)

The first positional argument is an expression string, parsed by **top-level colon count** — colons inside `[...]` or `(...)` do not count (`_count_colons_outside_brackets`):

| Colons | Form | Meaning | Source |
|---|---|---|---|
| 0 | `x` | 1-variable (distribution / histogram input) | `colon_count == 0` → 1D form (`drawer.py:852`) |
| 1 | `y:x` | 2-variable plot (y dependent, x independent) | `_split_top_level_colon` (raises if none) |
| 2 | `z:y:x` | 3-variable plot | `_split_top_level_colons_3` (Phase 13.39.DF) |

For the **positional y/x/z slots**, both raw `DFDraw` and the ADF surface resolve a **column name or a computed expression** — `_eval_column` returns `df[expr]` when it is a column, else `df.eval(expr)` (`drawer.py:984`). The **column-only** limitation applies to the **structural channels** `group_by`/`facet_by`, not to the positional slots (R-4 note; Phase 13.61 is the target that lifts that channel restriction). At the ADF surface the alias machinery additionally resolves aliases before delegating.

Worked example (executed): `draw("sqrt(y):log(x)", type="profile")` plots `df.eval('sqrt(y)')` against `df.eval('log(x)')`; standard `pandas.eval` functions (`sqrt`, `log`, `abs`, `exp`, and arithmetic) are available, and the axis labels default to the expression strings. Note the distinction the literature draws: this is a **data transform** (transform the values, then plot), not an **axis-scale transform** (keep the data, render the axis on a log scale) — the latter is **not** a [CURRENT] dfdraw option and is a separate candidate for Pass 2 (`xscale=`/`yscale=`).

### §2A.2 The y-slot vector — `[…]` (multiplicity)

A bracketed, comma-separated list in the y-slot is a **vector**: `[y0, y1, …]:x` → one curve per element. This is R-1 (scalar→vector) applied to the y-slot. Splitting is paren/bracket-aware, so `[max(a,b), c]:x` yields **two** elements (`_split_paren_aware`). Returns a **list** of per-element stats. [CURRENT, executed.]

- **Default layout = overlay on one axes.** The vector dispatch uses the caller's `same` for the first element and forces `same=True` for every subsequent element (`drawer.py:2108–2109`) → all elements share one axes.
- Elements are distinguished by **`vector_style`** — default `'color'`; when `group_by` is also set the default becomes `'linestyle'` (color reserved for groups).
- *Present-behavior boundary (Pass-2 seam, not a design statement):* a y-vector has **no per-element panel/facet layout** today — overlay is the only layout for it.

### §2A.3 Plot types — `type=`

Canonical type vocabulary (`_DRAW_TYPE_NAMES`, `drawer.py:293`): **`hist`, `scatter`, `profile`, `hist2d`, `profile2d`, `scatter3d`, `hexbin`**, by expression arity:

| Arity | Expression | Types |
|---|---|---|
| 1-variable | `x` | `hist` (distribution) |
| 2-variable | `y:x` | `profile`, `scatter`, `hist2d`, `hexbin` |
| 3-variable | `z:y:x` | `profile2d` (2D profile heatmap), `scatter3d` |

`type='profile2d'`/`'scatter3d'` require the 3-variable form (`drawer.py:4607–4611`, `:4657–4661`); `scatter3d` does not support `group_by` (`:4668`).

**Composition — `type='A+B'`** composes layer types left-to-right by z-order; the long form is `overlay(layers=[…])`. Both [CURRENT] on all four surfaces (R-5 table). `+` is the only composition operator.

### §2A.4 Channels (structural & aesthetic kwargs)

- **`group_by`** (+ `group_by_bins`, `group_by_quantiles`): partitions data into **curves on one axes** (one curve per group/bin). A continuous `group_by` requires `group_by_bins` or `group_by_quantiles` (cardinality guard). Expression-valued at the ADF surface (materialized before delegation); column-only at raw dfdraw (R-4 note). Group distinction channel default `'color'`.
- **`facet_by`** (+ `facet_by_bins`, `facet_by_quantiles`): partitions data into **subplots** (not curves). Dimensional grammar (`drawer.py:127–130`): `facet_by[0]` = **ROW**, `[1]` = **COL**, `[2]` = **FIGID** (separate figures), `[3+]` → `NotImplementedError`. `share` ∈ `{'all','row','col','none'}` controls sharex/sharey. The literal `facet_by='vector'` is a special non-column form (`drawer.py:3655`).
- **`color`**: a color channel — a column/expression mapped to colors, or a literal matplotlib color (e.g. `'red'`, `'#FF0000'`).
- **`selection`** (scalar) **+ `selection_vector`** (vector): row filter, a backend query string. Scalar applies globally; per-element folded by **logical AND** (R-2). Applied before binning/stats (so all downstream statistics, including per-bin quantiles, are on the selected rows).
- **`weights`** (scalar) **+ `weights_vector`** (vector): weighting; folded by **multiplication** (R-2).
- **`vector_compose`** ∈ `{'inner','outer'}`: how multiple vector kwargs combine. **Default `'inner'`** = element-wise pairing, equal lengths required (`drawer.py:1799`); `'outer'` = cross-product. The ADF surface auto-forces `'outer'` **only when `n_y == 1`** (single-Y) with a multi-element `selection_vector`/`weights_vector` (`AliasDataFrame.py:11409`/`:11461`, Phase 13.35); an explicit `vector_compose` is always respected, and **multi-Y composes `inner` (paired), not `outer`** — e.g. `y`-vector(4) × `selection_vector`(4) → 4 paired curves, not 16.

### §2A.5 Normalization — `normalize=` (+ `normalize_layout`)

Normalization folds **exactly two resolved curves** into a comparison (reference) panel. The load-bearing point — **omitted in rev a/b, the gap this rev corrects** — is *where the two curves come from*: they are the two elements of the **vector iteration**, computed by `_compute_vector_iteration_indices(n_y, selection_vector, weights_vector, vector_compose)` (`drawer.py:1607`, called at `:2475`; `len(indices) != 2` raises at `:2481`, parallel guards `:2789`/`:3110`). The two curves may therefore be sourced **interchangeably** from any vector channel — this is the basic functionality:

| Curve source | Form (vector order = numerator, denominator) | Status |
|---|---|---|
| **`selection_vector`** — the primary, tested form | `draw("y:x", type="profile", selection_vector=[sig, ref], normalize="ratio")` | [CURRENT], executed; the form used throughout `test_normalize.py` |
| **`weights_vector`** | `draw("y:x", …, weights_vector=[w0, w1], normalize="ratio")` (needs `vector_compose='outer'` when `y` is scalar — auto-forced by the ADF surface for single-Y only, `AliasDataFrame.py:11461`) | [CURRENT], executed |
| **y-vector** | `draw("[numerator, denominator]:x", type="profile", normalize="ratio")` | [CURRENT], executed |

This generality is exactly what a denominator-**naming** kwarg cannot express: `selection_vector[1]` and `weights_vector[1]` are not columns to be named — they are vector elements. A `normalize_ref='colname'` or `[a,b]/c:x` design addresses only the y-expression source and silently drops the selection- and weights-sourced cases that the test suite actually exercises. Modes (`drawer.py:2982–2987`):

| Mode | Comparison panel | Reference line |
|---|---|---|
| `delta` | numerator − denominator | 0 |
| `ratio` | numerator / denominator | 1 |
| `log_ratio` | ln(numerator / denominator) | — |
| `pull` | (numerator − denominator)/σ, with ±1σ/±2σ bands | 0 |

`normalize` also accepts a **callable** `f(stats_sig, stats_ref) → (values, errors)` (`Optional[Union[str, callable]]`, `drawer.py:4487`; `_compute_normalize_transform`). `normalize_layout` ∈ `{'overlay+diff'` (default, `:2514`), `'diff_only'` (`:2526`)`}`.

**Evidence (quoted, per the read-the-matrix-and-test requirement).** Source: `_compute_vector_iteration_indices` (`drawer.py:1607`/`:2475`) + the 2-curve guards (`:2481`/`:2789`/`:3110`). Tests: `tests/test_normalize.py` — `test_ND_*` (delta), `test_NR_*` (ratio), `test_NL_*` (log_ratio), `test_NP_*` (pull), `test_NC_*` (callable), `test_NLY_*` (layout), `test_NV_*` (count/validation), `test_NSC_*` (sign) — **every one sources the two curves from `selection_vector`**, confirming the vector-channel mechanism (not the y-vector form) is the tested contract.

*Present-behavior boundary (Pass-2 seam, corrected):* the current limit is **exactly two resolved curves, regardless of source**. The Pass-2 extension is to relax this to **N curves per source** (N numerators sharing one reference, preserving the vector-channel mechanism that already serves selection/weights/y uniformly) — **not** to introduce a denominator-naming syntax, which would discard the selection_vector/weights_vector sources.

### §2A.6 Statistical / rendering channels (descriptive)

- **`bins`**: bin count — scalar, or `[nx, ny]` for 2D types.
- **`range`**: axis range — `'minmax'`, explicit, or per-axis nested lists; when unset, an autorange strategy is chosen (e.g. `hybrid`, `robust_*`).
- **`central`** ∈ `{'mean','median','both','none'}`: profile central tendency (validated at `profile.py:381`). Default resolves to **`'mean'`** (`quantile.central_default`); `'both'` draws the mean and median lines simultaneously (`profile.py:904`); `'none'` draws no central line (band/error only, `:785`).
- **`quantiles`**: list of levels `[q…]` → per-bin quantile band, computed on the **selected (post-filter)** data, from the same arrays as the central statistic.
- **`fit`** / **`summary_fit`**: fit overlay + summary table; backend is `scipy.optimize.curve_fit` (dict tier inherits its vocabulary per P-5).
- **`min_entries`**: minimum per-bin entry count for a point to render.

### §2A.7 The three surfaces (R-7, concrete)

The same mini-language and channels apply identically across:

| Surface | Call form | Note |
|---|---|---|
| Interactive | `adf.draw(expr, **kwargs)` | the entry tier (P-0) |
| Figures | `adf.draw_figures([spec, …])` | each spec is a dict with `'expr'` + the same kwargs |
| Batch | `adf.draw_batch([spec, …])` | same spec vocabulary; routes through `draw()` (Phase 13.55) |

`'expr'` is required in every spec; composition strings and channels behave identically on all three (R-5 table; R-7).

### §2A.8 Output contract

Every draw returns `(fig, ax, stats)` (`DrawResult`, `drawer.py:54`). A y-vector returns `stats` as a **list**, one entry per element. `ax=` accepts a caller-supplied `Axes` (rendered into — this is how several plots are composed into one figure manually), and `same=True` overlays onto the current/last axes.

---

## §3 P-0 and the Four Principles

### P-0 — Interactive simplicity first (the RANKING principle)

> **Default interactive usage is simple: an expression plus a few switches —
> `tree->Draw`-like. Shortcuts and defaults are the primary interface; the
> full dictionary is the power tier, never the entry tier. A user at the
> prompt should never need an AI assistant (or this document) to make a
> standard plot.**

P-0 outranks P-1..P-4 and the grammar: they are means, P-0 is the end. Two
operational consequences:

1. **The approval gate.** Every interface change — new kwarg, fix option,
   standardization, deprecation — must answer first: *"did it make the
   user's life simpler?"* Standardization that does not benefit users is
   not implemented, even if it would improve internal consistency. When
   audit fix options conflict, the option preserving interactive simplicity
   wins.
2. **The simple-query test (acceptance criterion).** Given a dataset and
   `draw_help()` alone, a user produces each standard QA plot with an
   expression + at most a few switches — no dict, no documentation lookup.
   The corpus is the proof this is the real usage profile: 75 of 84
   production calls are `adf.draw(expr, type=…, selection=…, bins=…)`-class
   calls; dicts appear in ZERO of 84.

## §3a The Four Principles (restated; enforcement = the grammar; all subordinate to P-0)

| ID | Principle | Grammar binding |
|---|---|---|
| **P-1 Symmetry** | Every shortcut has a full form and they are equivalent | R-3 equivalence tests; R-4 family⇄dict identity |
| **P-2 Intuitive** | The obvious spelling works | R-6 aliases; §5 one vocabulary ('type' everywhere); R-2 folds match user intuition (AND / multiply) |
| **P-3 Full inline help** | The complete surface is discoverable at the prompt | declarations (§4) are runtime-introspectable; `draw_help` renders them (live, cannot drift) — **[TARGET]** dfdraw-side (13.57 deliverable; ADF-side exists since 13.56). Until it ships, declarations are enforced by the audit checklist alone |
| **P-4 Shortcut extends to full dict** | The dict is a strict superset of the shortcut | R-3 (primary key) + R-4 (family namespace); G-0 forbids the reverse direction ever breaking |

### P-5 — Backend pass-through (the dict tier inherits the evaluation library's vocabulary)

> *"If we use the full dictionary for some parameters, the content should be compatible with the library used for evaluation. … if we use fit, we should use parameters for the fit as in the minimizer. … I do not want to change too much in my interface. I am almost happy with what we have now, but I wanted to standardize the parts we can."*
> — M. Ivanov, 2026-06-11 (binding)

Beyond the dfdraw-owned dispatch key(s) (`'type'`, `'by'`, `'levels'`, `'format'` — the §5 list, kept minimal), the dict tier's keys are the underlying library's parameter names, passed through verbatim: `fit={'type':'gauss', 'p0':…, 'bounds':…, 'sigma':…, 'maxfev':…}` = `scipy.optimize.curve_fit` vocabulary. Consequences: (a) each §4 declaration names its **backend** and the pass-through namespace; (b) dfdraw never aliases or renames a backend parameter — `draw_help` and docstrings point at the backend's documentation for that namespace; (c) §5 stays a short list because everything else is inherited; (d) wrappers stay thin — minimal interface change, which is the architect's constraint. Serves P-0 directly: power users already know the backend's names.

Composability and round-trip principles from v1.0 remain **dropped** — constraint cost without a consumer asking. R-1/R-4 cover the composability that production actually uses.

---

## §4 Per-kwarg declaration schema (what each kwarg must ship)

```
kwarg:        selection
primary:      expr                                  # R-3
vectorizable: via selection_vector (axis: curves)   # R-1
entry_forms:  per surface — adf.draw: str | figures-spec: str | batch-spec: str   # R-1 caveat
fold:         AND                                   # R-2
family:       —                                     # R-4
aliases:      —                                     # R-6
backend:      pandas.eval (expression namespace)    # P-5
surfaces:     draw, figures-spec, batch-spec        # R-7
status:       per form — CURRENT (cite) / TARGET    # §2 legend
```

Eight lines per kwarg, ~20 kwargs (corpus §2 inventory). These declarations ARE the "worked example mapping" the architect required (*"I do not want to redo that repeatedly"*) — written once here, consumed by: the audit (expected-state column), the help system (P-3), and the equivalence-test generator (P-1). The full declaration set for the six audit-bound surfaces (`fit, summary_fit, time_format, quantiles, central, selection`) plus the R-4 families (`group_by_*, facet_by_*, quantiles_*`) is the first deliverable of the Phase 13.57.DF audit, validated against the corpus.

---

## §5 Canonical vocabulary

| Position | Canonical | Deprecated key-aliases (kept forever, warn) | Value-aliases (kept forever, silent) |
|---|---|---|---|
| "what kind of thing" | **`type`** (overlay precedent, already production) | `fun` (fit), `kind` (summary_fit) | `histo`→`hist`, `gaus`→`gauss` |
| family subject | **`by`** (inside `group_by`/`facet_by` dicts) | — | — |
| quantile levels | **`levels`** | — | — |
| format strings | **`format`** | — | — |

`kind` is reserved (never reused for a new meaning). Additions to any alias table require one line in this document's revision history — additions only, per R-6/G-0.

---

## §6 Back-compat: the executable conformance set

The §5 migration promise is checked, not trusted:

1. **Static census:** every §4 declaration must show its corpus-observed forms verbatim in a "runs as written" column. Hard pass/fail list (corpus §7): `fit='gauss'|'gaus'|'pol2'|'linear'`; `summary_fit='table'`; `time_format='%H:%M'`; `quantiles=[…]` (3/5/7-element); `central='median'`; `normalize='delta'|'ratio'|'pull'`; `type='hist2d+profile'|'histo'`; sibling families incl. expression-valued and list-valued `group_by`/`facet_by`; `range='minmax'` and nested lists; `on_error='skip'`; `selection`+`selection_vector` AND-composition.
2. **Executable:** the ADF gallery (42 figures) + full ADF suite run against any dfdraw change branch **before its phase closes** (cross-team gate; F-E precedent). The four production scripts re-censused with `extract_callforms.py` after any corpus revision.
3. **Simple-query test (P-0):** the fix phase ships a user-facing example set — a provided dataset + the standard QA queries, each written as expression + a few switches, runnable as-is. If any standard query needs a dict after the fix phase, that is a P-0 finding against the fix phase.

Any change that cannot satisfy both is `[BREAKING]`, requires an enumerated consumer-impact list (ADF call sites, gallery figures, production scripts), a dual-support deprecation period of at least one phase, and an explicit architect decision — never a default.

---

## §7 Enforcement

**New code (immediately on ratification):** a new shortcut/kwarg ships its §4 declaration + the R-3 equivalence test + docstring listing all keys + alias-table line — in the same commit. CRR includes a one-line P-0..P-5 conformance statement, **leading with the P-0 gate answer** ("did it make the user's life simpler?" — one sentence, falsifiable).

**Existing code (Phase 13.57.DF audit):** per kwarg, exactly two checks — (a) does the declaration match observed behavior (executed, not static; the v1 lesson), (b) do the generated R-3 equivalence tests pass. Violations become finding IDs with `[ADDITIVE]`/`[BREAKING]` flags. Input findings already filed: E-2 (`BUG_dfdraw_20260611_median_return_data`), E-3 (`…profile2d_ax_ignored`), E-4 (`…facet_by_ax_ignored`), F-B (named-param silent drop), F-C (StringDtype hist), the coder's six-surface drift probe. Fix strategy: three-option sheet per `[BREAKING]` item — **each option sheet answers the P-0 gate first**; an option that standardizes without simplifying is listed only with that disadvantage stated — §0 architect decisions, consolidated fix phase, independent coder (drafter pre-recusal honored).

---

## §8 Architect decisions (seven; panel recommendations + architect preliminary positions, 2026-06-11)

| # | Decision | Panel | Architect |
|---|---|---|---|
| D1 | Family dict key `'by'` (the dict is only an optional bundled spelling; flattened siblings stay primary) | CONFIRM (pandas-compatible; zero impact on existing calls) | **pending** |
| D2 | Future-pairs-only default fold = `replace-error` (existing AND/multiply folds [CURRENT], unchanged forever — RC-3 scope) | CONFIRM | agreed |
| D3 | Seed aliases: values `gaus→gauss`, `histo→hist`; keys `fun→type`, `kind→type`, **+ `initial→p0`, `guess→p0`** (fits.py:42, P-5 violations resolved by aliasing to scipy's name) | CONFIRM + EXTEND | OK |
| D4 | `quantiles_mode` prefixed alias (old `quantile_mode` kept forever) | CONFIRM | OK |
| D5 | `central='median'` end-to-end — **REVISED under P-0: scalar only, no dict form** (zero corpus demand). Fix = ADDITIVE `y_central` return_data column; `y_mean` unchanged; fit seeding already correct since 13.51 (5a closed; 5b = BUG_dfdraw_20260611_median_return_data fix direction). Corpus: used ×2 (ts_draw:187) | CONFIRM as revised | OK |
| D6 | Backend pin per surface (documentation-only: §4 declarations name the backend + pass-through namespace, e.g. `fit → scipy.optimize.curve_fit: p0, bounds, sigma, maxfev`) | per-P-5 audit deliverable; no code change | OK |
| D7 | Alias warn policy: value-aliases **silent, always**; key-aliases warn **once per Python session** (never per-call, never in CI) naming the canonical replacement — per-call warnings are a P-0 violation | CONFIRM (RC-4) | OK |

## §9 Ratification

> *(empty — architect fills on approval; quotes land verbatim per GP-3)*

## §10 Revision history

| Version | Date | Changes |
|---|---|---|
| 1.0 | 2026-06-11 | dfdraw-coder draft: principles without mechanics. Superseded; §0 narrative and P-1..P-4 names retained. |
| 1.1-draft rev f | 2026-06-11 | Panel corrections (Sonnet65 RC-1..RC-4 + Fable5_1 R-A..R-L): draw_help [TARGET] dfdraw-side; census re-stamped (84, hashed set); D2 scope sentence; §8 → seven decisions (D5 scalar-only/additive y_central; D6 documentation-only; D7 warn-once; D3 + initial/guess→p0); current dict keys stated; [TARGET]=P-0-gated candidates; group_by expressions = ADF-layer contract; R-5 per-surface table; §4 schema + entry_forms/backend/status; R-K clash example; §0a Lineage; minimal-change quote (R-L). |
| 1.1-draft rev e | 2026-06-11 | P-5 Backend pass-through added (architect, binding quote): dict tier inherits the evaluation library's parameter names beyond the minimal dfdraw dispatch keys; §4 declarations gain a backend field; §5 stays minimal by inheritance. |
| 1.1-draft rev d | 2026-06-11 | P-0 (Interactive simplicity first) added as the RANKING principle with 4 new binding quotes; approval gate in §7 (CRR statement + [BREAKING] option sheets answer P-0 first); §6 simple-query test added; corpus datum: dicts appear in 0 of 84 production calls. |
| 1.1-draft rev c | 2026-06-11 | §1 quote cleanup at architect request (my_times sentence removed — unverifiable origin; v1.0-verdict quote removed — context-free).  Panel fix (fable5_5 [!], 3 executed counter-examples): CURRENT/TARGET tagging on every grammar example; ';' expr spelling re-attributed (architect-required need, NOT a current form — no corpus instance; ValueError at HEAD); R-1 per-surface entry-form caveat; R-2 AND-fold source citation (drawer.py L1480/L1809). |
| 1.1-draft | 2026-06-11 | Consolidated single document (rev b: quote spelling normalized at architect request; companion references updated to the 3-document package): + §1 architect verbatim quotes (binding); + §2 grammar G-0/R-1..R-7 (reverse-engineered from the 84-call corpus); + §4 declaration schema; + §5 vocabulary with append-only aliases; + §6 executable conformance set; P-5/P-6 dropped. Drafted by Fable1 (ADF) at architect request for dfdraw-team ratification. |
| 1.2-draft | 2026-06-21 | + §2A Grammar of Graphics: descriptive **[CURRENT]** specification of the as-built grammar (expression mini-language & colon arity; y-slot vector + overlay-default layout; type vocabulary by arity + `A+B` composition; channels `group_by`/`facet_by` row/col/figID/`color`/`selection`/`weights`; normalization 2-curve modes; statistical/rendering channels; the three surfaces; output contract). Consolidates behavior previously only in `drawer.py` + docstrings. **Pass 1 — descriptive only**; no new/changed behavior; all forms source-cited against canonical `drawer.py` 8,082 / MD5 `e851acc7…`. Extension design (y-vector layout; N-numerator normalization) deferred to Pass 2 brainstorming. Drafted by Opus48_3 at architect request. |
| 1.2-draft rev c | 2026-06-21 | §2A.5 **normalization curve-source mechanism corrected** (the real gap; verified by execution + source + tests). Rev a/b described normalization as a "2-curve y-vector `[num, denom]:x`" only — **wrong/incomplete**: the two curves are the elements of the **vector iteration** (`_compute_vector_iteration_indices` `drawer.py:1607`/`:2475`; 2-curve guards `:2481`/`:2789`/`:3110`) and may be sourced interchangeably from **`selection_vector`** (the primary, tested form — every test in `tests/test_normalize.py`), **`weights_vector`**, or the **y-vector**. Executed: all three render; `selection_vector` and y-vector at default compose, `weights_vector` with `vector_compose='outer'`. Functionality and tests were never missing — only the §2A description was, which is why the Pass-2 panel proposed `normalize_ref='c'`/`[a,b]/c` (an expression-only denominator-naming syntax that cannot reference a selection- or weights-sourced curve and would remove the tested capability). Pass-2 seam re-scoped: relax "exactly 2 resolved curves" to "N per source", **keeping** the vector-channel mechanism — not a denominator-naming syntax. Also added (executed, [CURRENT]): §2A.1 expression-in-slot worked example (`sqrt(y):log(x)` via `_eval_column`) + the data-transform vs axis-scale-transform distinction (`xscale=`/`yscale=` is the real gap, deferred). Drafted by Opus48_3; all claims re-verified against canonical drawer.py + executed against tests/test_normalize.py. |
| 1.2-draft rev d | 2026-06-27 | **§2A `vector_compose` default corrected (documentation bug, source-verified).** Rev a–c stated the `vector_compose` "default `'outer'` at the ADF surface" (R-1 composition note; §2 channel list; §2A.5 weights row) — **wrong**. The core default is **`'inner'`** (element-wise pairing, `drawer.py:1799`); the ADF surface auto-forces `'outer'` **only for single-Y (`n_y == 1`) + a multi-element `selection_vector`/`weights_vector`** (`AliasDataFrame.py:11409`/`:11461`, Phase 13.35), respecting any explicit `vector_compose`. A multi-element **Y-vector composes `inner` (paired)** — e.g. `y`-vector(4) × `selection_vector`(4) → **4 paired curves, not 16**. The error was an over-summarization of the Phase 13.35 single-Y auto-force into a blanket "outer default"; the correct, conditional behavior is stated plainly in the ADF docstring (`AliasDataFrame.py:11409–11430`: "No-op if expr has >1 Y expressions — multi-Y handles inner natively"). **Process note:** the wrong claim cited a *phase* (13.35), not a *source line*; `[CURRENT]` claims must cite source lines so the condition is visible at write time. Same failure class as the §2A.5 incident (adjacent §2A claim not re-verified when §2A.5 was corrected in rev c). Drafted by Opus48_3 at architect request; verified against canonical `drawer.py:1799` + `AliasDataFrame.py:11409/11461`. |
| 1.2-draft rev b | 2026-06-21 | §2A panel corrections (Sonnet65 ×9, all re-verified against canonical): **P1-A** `central='both'` added to §2A.6 (`profile.py:381` validation, `:904` draws both lines); **P1-B** zero-colon `x` row added to §2A.1 table + positional-slot claim corrected (raw `DFDraw` resolves `df.eval` expressions in y/x/z via `_eval_column` `drawer.py:984`; the column-only limit is scoped to structural channels `group_by`/`facet_by`); **P2** line-number fixes (`_DRAW_TYPE_NAMES` :294→:293; normalize modes :2987→:2982–2987; `facet_by='vector'` +`:3655`); **P3** completeness (`normalize_layout` `'overlay+diff'`/`'diff_only'` :2514/:2526; `normalize=` accepts a callable :4487). **Naming vote** (10 reviewers, unanimous NO on "Grammar of Graphics and Statistics"): kept "Grammar of Graphics" + subtitle "— expressions, statistical transforms, and layout" + Wilkinson scope sentence (statistics is component #4, not an add-on); deeper ggplot2/Wilkinson comparison deferred to `dfdraw_COMPARISON.md`. Literature research postponed (does not block Pass 1). |
