# Sampling Technical Summary

**Subproject:** sampling (Phase 13.10.DF – 13.11.DF)  
**Path:** `UTILS/dfextensions/sampling/`  
**Last updated:** 2026-05-18  
**Source version:** commit `83974203` (run_tests.sh)

---

## Purpose

Threshold-based downsampling with PDF estimation for steeply falling physics distributions (pT spectra, dE/dx, occupancy). Preserves spectral shape via correction weights while reducing dataset size by 10–100×.

---

## Public API

### Core Functions (downsample.py)

| Function | Purpose | Signature |
|----------|---------|-----------|
| `downsampleDF` | Binned groupby downsampling | `(df, variables, frac, random_state, *, mask=None)` |
| `downsampleDFTrigger` | Multi-trigger bitmask downsampling | `(df, variables, triggers, random_state, *, mask=None)` |
| `downsampleDFSmoothFactorized` | Factorized smooth PDF (product of 1D marginals) | `(df, variables, random_state, *, frac=None, threshold=None, pdf_func=None, pdf_params=None, mask=None, debug=False)` |
| `downsampleDFSmooth` | Full ND joint PDF (D≤5, scipy) | `(df, variables, random_state, *, frac=None, threshold=None, pdf_params=None, mask=None, debug=False)` |
| `downsampleDFSmoothTrigger` | Multi-trigger with smooth PDF | `(df, variables, triggers, random_state, *, pdf_params=None, mask=None)` |

### Key Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `variables` | dict | `{name: (n_bins, lo, hi)}` (Option C) or `{name: edges_array}` (Option D) or `{name: 'categorical'}` |
| `frac` | float | Target sampling fraction (0, 1]. Mutually exclusive with `threshold`. |
| `threshold` | float | Direct threshold. Mutually exclusive with `frac`. |
| `pdf_func` | callable | Optional analytical PDF. Bypasses empirical estimation. |
| `pdf_params` | dict | `{"poly_order": 2, "poly_half_range": 0.5, "fit_coordinate": "x"|"bin"}` |
| `mask` | str or array | Selection mask. Affects both PDF estimation and sampling (AD-8). |

### Output

All functions return a sampled DataFrame with added columns:
- `_pdf`: estimated PDF at each point
- `_threshold`: threshold value used
- `is_sampled` (in scan tree): 0/1 sampling indicator
- Correction weight: `1/max(pdf, threshold)` (derivable, not stored)

---

## Sampling Algorithm

```
Accept x_i if: pdf(x_i) × U_i < threshold    (U_i ~ Uniform(0,1))
Weight: cw_i = 1/max(pdf(x_i), threshold)
```

- **Dense regime** (pdf > threshold): acceptance = threshold/pdf, weight = 1/pdf
- **Sparse regime** (pdf ≤ threshold): acceptance = 1, weight = 1/threshold (capped)
- `frac → threshold` via bisection (~60 iterations)
- N_eff/N_sampled ≈ 44% for Gaussian frac=0.1

---

## PDF Estimator (v5, 3-Layer)

1. **Gaussian kernel smoothing** (σ = 0.5 × Δx)
2. **Poisson plug-in correction** × (1 − e^(−n_bin))
3. **Local polynomial regression** (order 2, ±0.5× half-range)

For non-uniform bins (`fit_coordinate="bin"`):
- Layer 3 fits counts vs bin-index (not pdf vs x)
- Interpolation in bin-index space
- Edge bins: log-linear extrapolation to bin edges

Achieves <1% bias for pdf_true > 0.01 across Δx = 0.05/0.1/0.2.

---

## Validation Framework

### Scan Infrastructure

- `generate_scan_tree.py`: 1000 iterations, random (N, frac, Δx/nbins)
- `validate_scan.py`: S1–S4c figures + summary report (distribution-agnostic)
- `run_tests.sh`: parallel 3-config regression suite

### Figures

| Figure | What it tests | Key metric |
|--------|--------------|------------|
| S1 | Sampling fraction accuracy | ⟨y⟩ ≈ 0, σ ∝ 1/√(N×frac) |
| S2 | Threshold stability | σ(thr) vs 1/√(N×frac) |
| S3 | PDF estimator bias | RMS = √(a²/λ + b²), a ≈ 0.5 (v5) |
| S3b | Bias vs position/dx | Edge vs non-edge split |
| S4a | Spectra recovery by frac | Unbiased ratio ≈ 1.0 |
| S4b | Spectra recovery by Δx | σ model agreement |
| S4c | σ scaling validation | slope ≈ 1.0, R² > 0.95, pull RMS ≈ 1.0 |

### Validated Configurations

| Config | S4c slope | S4c R² | Spectra ratio |
|--------|-----------|--------|---------------|
| Gaussian uniform | 1.00 | 0.96 | 0.9996 |
| Gaussian quantile (fit_coordinate=bin) | 1.00 | 0.97 | 1.0001 |
| Linear uniform | 0.98 | 0.97 | 0.9979 |

---

## File Inventory

| File | Lines | Purpose |
|------|-------|---------|
| `downsample.py` | ~1400 | Core downsampling functions + PDF estimator |
| `generate_scan_tree.py` | ~450 | Scan tree generation (Gaussian/linear, uniform/quantile) |
| `validate_scan.py` | ~1600 | Scan validation + S1–S4c figures |
| `validate_statistical.py` | ~1200 | Legacy T1–T8 fixed-point tests |
| `perf_figure_threshold.py` | ~250 | Performance figures (3 distributions × 2 PDF modes) |
| `run_tests.sh` | ~180 | Parallel regression test suite |
| `tests/test_sampling.py` | ~180 | 17 binned downsampling tests |
| `tests/test_sampling_smooth.py` | ~400 | 36 smooth factorized + ND tests |
| `tests/test_smoke_pdf_estimator.py` | ~450 | 22 PDF estimator smoke tests (S1–S13) |

---

## Dependencies

- numpy, pandas, scipy (gaussian_filter1d, stats)
- matplotlib (figures only)
- uproot (scan tree I/O)
- No ROOT dependency in core `downsample.py`

---

## Known Limitations

- Edge bins in quantile binning have higher bias (RMS ~0.14 vs ~0.01 core). Mitigated by log-linear extrapolation but not eliminated for nbins < 50.
- ND smooth estimator (`downsampleDFSmooth`) limited to D ≤ 5 by scipy histogram.
- `fit_coordinate="bin"` only implemented for factorized path, not full ND.
- No 2D scan validation yet (1D only). 2D proposal drafted.

---

## Next Steps

1. **2D factorized validation** — pT (exponential) × multiplicity (power law), uniform vs quantile
2. **README.md** — user documentation with recipes
3. **Integration with ADF** — `aDF.draw(..., weights='correction_weight')` pipeline verified
