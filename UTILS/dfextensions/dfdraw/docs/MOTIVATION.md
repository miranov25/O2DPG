# MOTIVATION.md — dfextensions / dfdraw Project Motivation

**File:** MOTIVATION.md
**Location:** `docs/MOTIVATION.md`
**Maintainer:** M. Ivanov (architect)
**Created:** 2026-06-09
**Status:** Living document — update as new quantifications arrive or scope expands.
**Cross-references:** `ARCHITECT_DECISIONS.md` (GP-1..GP-9), `dfdraw_Technical_Summary.md` §Executive Summary

---

> This file is the canonical, grep-able, permanently tracked source for why this project
> exists. Other documents (TS, README, comparison doc) cite this file rather than
> carrying the motivation inline. It is not append-only — update it when numbers
> improve or the scope changes.

---

## 1. The Thesis

> **"Understanding is prerequisite for high-quality data."**
> — M. Ivanov, dfextensions presentation, December 2025

> *"Most important is the lack of understanding. Worse resolution is just a consequence.
> Good software → understanding → results close to the intrinsic resolution.
> Bad software → total lack of understanding."*
> — M. Ivanov, dfdraw governance session, 2026-06-05

Resolution is a consequence; understanding is the cause. Calibration QA at LHC physics
scale is not a plotting problem — it is the workflow that determines whether the
detector's intrinsic resolution can be recovered or not.

---

## 2. Why This Stack Exists

### 2.1 The Run 1/2 baseline

ALICE Run 1 and Run 2 calibration QA was an **analyst-bandwidth** workflow — by design,
not by accident. The analyst-facing layer was a downsampled queryable dataset produced
continuously as part of production: data was either written in queryable format or
actively cached and downsampled (typically O(10⁻⁴) sampling, chosen to make
distributions roughly flat in the variables of interest). `TTree::Draw` queries ran in
seconds against this layer. A physicist asked a calibration question —
`TTree::Draw("dca_y:tgl:sector", "...", "prof")` — and saw the result. Hypotheses
tested per day was bounded by analyst thinking speed, not by the framework.

The downsampling code itself was production infrastructure. Without it, declarative
queries against full reconstruction data would have been impossible even in Run 1/2.
This is a general pattern, not specific to HEP: large-scale data systems serve
production and analysis through separate but coordinated layers; declarative
queryability lives at the analysis layer, not the production layer.

### 2.2 The Run 3 regression

Run 3's O2 / O2Physics framework was optimized for streaming production throughput;
the analyst-facing downsampled layer was not included in the initial framework design.
The consequences:

1. Calibration conditions are stored in CCDB as keyed binary blobs without a standard
   query interface, requiring bespoke C++ to read and interpret each one.
2. Each new calibration question requires bespoke imperative code — written, compiled,
   debugged, validated.
3. The cognitive cost of formulating a question shifted from the analyst (declarative
   one-liner, seconds) to the developer (imperative code, hours to days). Calibration
   QA became **developer-bandwidth-bound**, not analyst-thought-bound.

Production throughput and analyst-facing declarative query are two different design
points serving different purposes; they are not substitutable. The industry-wide return
from blob-store architectures to declarative query languages over the past decade —
Snowflake, BigQuery, Databricks SQL, RDataFrame in ROOT, modern HEP analysis
ecosystems — reflects the same lesson.
*(Stonebraker & Pavlo, "What Goes Around Comes Around... And Around", 2024.)*

The three failure modes of the traditional Run 3 approach:

1. **New calibration → recompile C++ → days of delay.** Hypotheses are expensive;
   few are tested per cycle.
2. **Debugging → gdb, print statements → limited insight.** Each bug investigation
   is a custom project, not a query.
3. **Understanding data → write custom code → error-prone, often impossible.** The
   cognitive load is dominated by the cost of writing the code, not by the question
   itself.

### 2.3 The foundational principle (GP-9)

> *"I do not trust custom code — I trust standard interactive queries and derived
> statistics. Validation is a statistical process requiring flexibility and
> interactivity."*
> — M. Ivanov, dfextensions presentation, December 2025

See `ARCHITECT_DECISIONS.md` GP-9 for the ratified governance entry.

The queryability gap is being addressed from two architectural levels in parallel:
the O2CCDBAI initiative addresses the calibration-DB storage layer, and the
dfextensions stack addresses the analysis layer. The downsampled-dataset pattern from
Run 1/2 continues: `examples/time_series.py` is precisely such a downsampled dataset,
produced at production scale, and dfextensions provides the declarative grammar over it.
The two efforts together continue the dual-layer architecture pattern that worked in
Run 1/2, adapted to Run 3 data.

---

## 3. The Measurable Consequence

The productivity collapse in calibration QA has a measurable downstream effect on
physics. **σ(pT) / pT degrades by a factor of approximately ×2.4 in Run 3 relative
to the best Run 2 calibration.** The detector's intrinsic resolution has not changed;
the workflow's ability to converge to it has.

> This is the only quantification officially approved for citation as of
> `ARCHITECT_DECISIONS.md` AD-2/TS_v6.DF. Further quantifications (μm-level
> alignment numbers, ITS/TRD residuals) will be added here as they are confirmed
> from real-data validation.

The stack is domain-general: calibration, alignment, simulation / MC data remapping,
and physics analysis share the same grammar and return contract.

---

## 4. The Discovery Goal

> *"By extracting multidimensional differential maps, we can obtain maps that are
> already analytically tractable. That is usually our goal — to obtain a
> multidimensional function that we can understand and, ideally, also describe with
> an analytical model or analytically derived effective parameterization."*
> — M. Ivanov, dfextensions presentation, December 2025

This is the inverse of opaque-model workflows. Each fit produces parameters that can
be inspected; each correction reduces to coefficients with physical meaning; each
iteration is a step toward an analytically tractable description rather than a more
complex pipeline.

> *"Goal: production-ready composable tools with C++ speed."*
> — M. Ivanov, dfextensions presentation, December 2025

---

## 5. The Epistemological Question

This stack answers a specific question:

**Can production throughput and differential understanding coexist?**

The answer is yes — through the dual-layer architecture pattern. Production runs at
full scale and writes queryable downsampled data. The dfextensions stack provides the
declarative grammar over that downsampled layer. The two layers serve different
purposes and different timescales; neither is a substitute for the other.

---

## 6. The Stack Identity

dfextensions is an integrated three-component stack. No single component alone covers
the workflow:

1. **AliasDataFrame (ADF)** — data layer. Declarative derived columns (aliases) defined
   once, referenced anywhere. Subframe joins (e.g., GB-regression coefficient frames
   register as subframes). Lazy evaluation is essential: multi-TB tracking datasets
   cannot be flat-materialized into pandas. ADF resolves only the columns each plot
   needs, and only when the plot is drawn.

2. **GBregression** — analysis layer. Multi-dimensional parallel fits, sliding-window
   fits, prediction registration. Outputs are coefficient DataFrames consumed as ADF
   subframes. N-D decomposition via ADF subframes + aliases is one of the most
   important functionalities of the stack.

3. **dfdraw** — visualization layer. Declarative composition grammar + per-bin
   statistics + inline fits + differential operations (`normalize=`,
   `selection_vector=`, `weights_vector=`). Output is `(fig, ax, stats)` — `stats`
   is structured and consumable downstream.

Together this stack is analogous to R's *ggplot2 + broom + lme4* integrated
ecosystem — with no clean public Python equivalent. Standard composable tools rather
than custom code; declarative queries rather than imperative scripts; statistics that
flow back as data rather than locked behind plot rendering.

---

## 7. Scope

Current production use cases:
- TPC calibration QA
- ITS/TRD alignment
- Simulation / MC data remapping
- Multiplicity studies
- Time-series QA (calibration monitoring)

Cross-team audiences: ADF team, GBregression team, O2DistAI team, TimeAI team,
calibration teams.

---

## 8. Version History

| Date | Change |
|------|--------|
| 2026-06-09 | File created from TS v6.3 Executive Summary + session material (GP-3 verbatim quotes preserved). Sonnet65 drafting from committed TS content + governance session records. |

---

*docs/MOTIVATION.md — dfextensions / dfdraw*
*Canonical motivation source. Other documents cite this file.*
*GP-3 compliance: all architect quotes preserved verbatim including original phrasing.*
