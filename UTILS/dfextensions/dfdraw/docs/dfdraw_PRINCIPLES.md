# dfdraw_PRINCIPLES.md — Design Principles & Shortcut Grammar

**Version:** 1.1-draft (consolidated: principles + grammar + vocabulary in ONE document)
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

Composition of several vector kwargs in one call follows the existing `vector_compose` semantics (`'outer'` default at the ADF surface, Phase 13.35) — [CURRENT]; the grammar names what is already implemented.

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
