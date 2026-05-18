# PHASE_HISTORY.md — Sampling Subproject (Phase 13.10.DF – 13.11.DF)

**Subproject token:** sampling  
**Repository:** O2DPG, branch `feature/groupby-optimization`  
**Path:** `UTILS/dfextensions/sampling/`  
**Main Architect:** Marian (miranov25)  
**Main Reviewer (dfdraw):** Claude40  
**Reviewer 31 (sampling):** Claude (this session)

---

## Phase 13.10.DF — Downsampling Module Port (CLOSED)

| Version | Date | Commit | Summary |
|---------|------|--------|---------|
| v3.0 | 2026-03-15 | `22e99a8` | Port `downsampleDF` + `downsampleDFTrigger` from distortionMLFit/pidSkimmedFit. 16 tests. |
| v3.1 | 2026-03-15 | `a17a920` | Add `downsampleDFSmoothFactorized` + `downsampleDFSmooth`. Log-space empty-bin interpolation. 38 tests. |

## Phase 13.11.DF — PDF Estimator + Sampling Algorithm (IN PROGRESS)

### v1.0–v1.1: Interface Redesign (CLOSED)

| Version | Date | Commit | Summary |
|---------|------|--------|---------|
| v1.0 | 2026-03-16 | `cce8ed2` | Variable dict (Option C/D/categorical), mask parameter, trigger function. 52 tests. |
| v1.1 | 2026-03-16 | `cce8ed2` | 4-reviewer consolidated review. APPROVED. |

### v2.0–v2.1: Threshold Sampling + v5 Estimator (CLOSED)

| Version | Date | Commit | Summary |
|---------|------|--------|---------|
| v2.0 | 2026-03-18 | `8a4879c` | Debug columns, binned PDF with Poisson correction. 52 tests. |
| v2.1 | 2026-03-18 | `8d380d1` | 3-layer PDF estimator (kernel + Poisson + polynomial). 67 tests. |
| v2.1 | 2026-03-18 | `527eb30` | Threshold sampling algorithm, perf figures, chi² fix. 75 tests. |
| v2.1 | 2026-03-18 | `80d299b` | AD-14: production _pdf/_threshold columns. T1/T2 validation. |

### v2.2: Parameter Scan Validation (CLOSED)

| Version | Date | Commit | Summary |
|---------|------|--------|---------|
| v2.2 | 2026-03-19 | `0bf47f7` | `generate_scan_tree.py` + `validate_scan.py`. 1000-iteration scan. S1–S4c validated: slope=1.00, R²=0.96. |
| v2.2 | 2026-03-19 | `a7181fe` | S4 per-x-bin spectra recovery (S4a by frac, S4b by Δx). |
| v2.2 | 2026-03-19 | `a89c8c4` | S4c σ model: RMS(σ_model_i²), pull formula. Pull RMS ≈ 1.0. |

### v2.3: Quantile Binning + Linear Distribution + Distribution-Agnostic (CLOSED)

| Version | Date | Commit | Summary |
|---------|------|--------|---------|
| v2.3 | 2026-03-20 | `8dc6d38` | S3b figure, S2/S3 dx_bin mismatch fix. |
| v2.3 | 2026-03-20 | `1835049` | `fit_coordinate="bin"` (counts vs bin-index), isEdge/nBins columns. |
| v2.3 | 2026-03-20 | `d6fded1` | S3b 2×4 layout, S2 edge-aware subset. Edge-excluded tail RMS: 0.054. |
| v2.3 | 2026-03-20 | `66fa30a` | S3 edge-excluded variant. R² jumps 0.19 → 0.87 excluding edges. |
| v2.3 | 2026-03-21 | `f8ab5ce` | Log-linear edge extrapolation. Tail RMS: 1.29 → 0.62. |
| v2.3 | 2026-03-21 | `6fbc2ea` | Distribution-agnostic `validate_scan.py`. Linear ramp f(x)=(1+0.1x)/10. |
| v2.3 | 2026-03-21 | `e42a91d` | Fix S8 test for edge-extended pdf. |
| v2.3 | 2026-03-23 | `85b22ce` | Fix analysis range regression (P0). `--x_range` CLI + params x_lo/x_hi. |
| v2.3 | 2026-03-23 | `8397420` | `run_tests.sh`: parallel 3-config regression suite. |

### v2.4: 2D Factorized Validation (PENDING)

Proposal drafted (`PHASE_13_11_DF_v2_3_Proposal.md`). pT (exponential) × multiplicity (power law). Two binning strategies: uniform vs CDF-quantile. Awaiting architect approval.

---

## Key Architectural Decisions

| AD | Decision | Phase |
|----|----------|-------|
| AD-11 | Empty bins: PDF=0 for binned path | v2.0 |
| AD-14 | `_pdf`, `_threshold` always in output (production, not debug-only) | v2.1 |
| AD-15 | Both binned and polynomial estimator columns in scan tree | v2.2 |
| — | Per-point dx, not mean dx (4× coder override attempt) | v2.3 |
| — | `fit_coordinate="bin"`: fit counts vs bin-index for non-uniform bins | v2.3 |
| — | Edge extrapolation: extend polynomial evaluation to bin edges | v2.3 |
| — | Distribution-agnostic validation: meta from data, not hardcoded SIGMA | v2.3 |

---

## Key Incidents

| Incident | Impact | Rule/Fix |
|----------|--------|----------|
| Per-point dx override (4×) | Coder substituted mean dx for per-point dx, overriding architect | P0 governance escalation; confirms MUST when fighting training-data bias |
| S2/S3 dx_bin mismatch | Double `pd.qcut` with mismatched bin edges → empty panels | Fix: use global bin assignment, not per-method qcut |
| Gaussian range regression | `derive_scan_metadata` used data extremes (±5.7σ) instead of ±3σ | Store x_lo/x_hi in params table; `--x_range` CLI override |
| Coder hardcoded SIGMA | Distribution-specific code where generic was required | Refactored to `pdf_true_at()` from scan tree data |
| Git amend with children | Coder amended pushed commit, orphaning child | Rule 11 added to Coder QRC |

---

## Test Infrastructure

| File | Tests | Coverage |
|------|-------|---------|
| `tests/test_sampling.py` | 17 | Binned downsampling (groupby, trigger) |
| `tests/test_sampling_smooth.py` | 36 | Smooth factorized + ND |
| `tests/test_smoke_pdf_estimator.py` | 22 | S1–S13 PDF estimator smoke tests |
| `run_tests.sh` | 3 configs | Gaussian uniform, Gaussian quantile, linear (parallel) |

**Total: 75 unit tests + 3 integration scan configurations.**

---

## Scan Configurations (validated)

| Config | Distribution | Binning | fit_coordinate | Key result |
|--------|-------------|---------|----------------|------------|
| Gaussian uniform | N(0,1) | uniform Δx∈[0.05,0.2] | x | S4c: slope=1.00, R²=0.96 |
| Gaussian quantile | N(0,1) | CDF-quantile, nbins∈[20,500] | bin | S4c: slope=1.00, R²=0.97 |
| Linear | (1+0.1x)/10 | uniform Δx∈[0.05,0.2] | x | S3: a=0.84, R²=0.98 (pure Poisson) |
