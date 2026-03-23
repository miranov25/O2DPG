#!/usr/bin/env python
"""
generate_scan_tree.py

Generate parameter scan trees for downsampling validation.

Runs 1000 iterations with RANDOM (N, frac, Δx) per iteration.
Validates theoretical scaling laws across 2 decades in N.

Scan parameters (per iteration):
  N:    log-uniform in [10^4, 10^6]  (flat in log-space = equal per decade)
  frac: uniform in [0.01, 0.1]
  Δx:   uniform in [0.05, 0.2]

Methods: smooth (legacy), smooth_v5 (3-layer)
Both use threshold sampling → _pdf, _threshold, _is_sampled stored.
Weights NOT stored (derivable: w = 1/max(pdf, thr)).

Output tree: 'scan' — sampled rows only (is_sampled=1 for at least one method).
Per-iteration metadata stored as constant columns: N, frac, dx.

Column naming:
  Common:  iteration (int16), x (float16), pdf_true (float32),
           N (int32), frac (float16), dx (float16)
  Per method: {method}_pdf (float16), {method}_threshold (float16),
              {method}_is_sampled (int8)

Usage:
    python generate_scan_tree.py [--n_iter 1000] [--output scan_validation.root]
    python generate_scan_tree.py --n_iter 10  # quick test

References:
    PHASE_13_11_DF_v2.2_Brainstorming.md §4
    AD-14: threshold as production column
    AD-15: both estimator columns (smooth + smooth_v5)
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import argparse

# Import from the downsample module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dfextensions.sampling.downsample import downsampleDFSmoothFactorized

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
    print("WARNING: uproot not available, will save as CSV instead")


# ===================================================================
# Configuration
# ===================================================================

SIGMA = 1.0
RANGE = 6  # ±6σ

# Scan ranges (per §4.2 of brainstorming v2.2)
N_MIN = 10_000       # 10^4
N_MAX = 1_000_000    # 10^6
FRAC_MIN = 0.01
FRAC_MAX = 0.10
DX_MIN = 0.05
DX_MAX = 0.20

# v5 estimator params (fixed — validated in Phase 13.11.DF v2.1)
PDF_PARAMS_V5 = {"poly_order": 2, "poly_half_range": 0.5, "kernel_sigma_bins": 0.5}


# ===================================================================
# Helper functions
# ===================================================================

def gaussian_pdf(x, sigma=SIGMA):
    """Analytical Gaussian PDF."""
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


# Linear ramp: f(x) = (1 + slope*x) / L on [-L/2, L/2], normalized
LINEAR_HALF = 5.0   # support [-5, 5]
LINEAR_SLOPE = 0.1  # mild slope

def linear_pdf(x):
    """Linear ramp PDF: (1 + 0.1*x)/10 on [-5, 5]."""
    x = np.asarray(x, dtype=np.float64)
    L = 2 * LINEAR_HALF
    return np.where((x >= -LINEAR_HALF) & (x <= LINEAR_HALF),
                    (1.0 + LINEAR_SLOPE * x) / L, 0.0)


def generate_data(rng, N, distribution="gaussian"):
    """Generate N random samples from the specified distribution."""
    if distribution == "linear":
        # Inverse CDF sampling for f(x) = (1 + slope*x)/L on [-a, a]
        # CDF(x) = (x + a)/L + slope*(x² - a²)/(2*L)
        # Use rejection: envelope = max_pdf = (1 + slope*a)/L
        a = LINEAR_HALF
        L = 2 * a
        max_pdf = (1.0 + LINEAR_SLOPE * a) / L
        samples = []
        while len(samples) < N:
            batch = N - len(samples) + 1000
            x_cand = rng.uniform(-a, a, batch)
            u = rng.uniform(0, max_pdf, batch)
            accept = u < (1.0 + LINEAR_SLOPE * x_cand) / L
            samples.extend(x_cand[accept])
        return np.array(samples[:N])
    else:
        return rng.normal(0, SIGMA, N)


def true_pdf(x, distribution="gaussian"):
    """Analytical PDF for the specified distribution."""
    if distribution == "linear":
        return linear_pdf(x)
    else:
        return gaussian_pdf(x)


def draw_scan_params(rng, mode="uniform", nbins_min=20, nbins_max=500):
    """Draw random (N, frac, Δx/nbins) for one iteration.

    N:    log-uniform in [N_MIN, N_MAX] — equal representation per decade
    frac: uniform in [FRAC_MIN, FRAC_MAX]
    Δx:   uniform in [DX_MIN, DX_MAX] (uniform mode)
    nbins: uniform int in [nbins_min, nbins_max] (quantile mode)
    """
    log_n = rng.uniform(np.log10(N_MIN), np.log10(N_MAX))
    N = int(10 ** log_n)
    frac = rng.uniform(FRAC_MIN, FRAC_MAX)
    if mode == "quantile":
        nbins = rng.randint(nbins_min, nbins_max + 1)
        dx = 0.0  # placeholder — per-point dx computed in iteration
        return N, frac, dx, nbins
    else:
        dx = rng.uniform(DX_MIN, DX_MAX)
        return N, frac, dx, 0


def run_one_iteration(iteration, N, frac, dx, pdf_params_v5=None, distribution="gaussian"):
    """Run one scan iteration with given parameters.

    Returns DataFrame with sampled rows only (union of both methods).
    Columns: iteration, x, pdf_true, N, frac, dx,
             smooth_pdf, smooth_threshold, smooth_is_sampled,
             smooth_v5_pdf, smooth_v5_threshold, smooth_v5_is_sampled
    """
    rng_data = np.random.RandomState(iteration)
    half_range = LINEAR_HALF if distribution == "linear" else RANGE
    lo, hi = -half_range, half_range
    n_bins = max(4, int(2 * half_range / dx))

    # Generate data
    x = generate_data(rng_data, N, distribution=distribution)
    df = pd.DataFrame({"x": x})

    # Filter to range (same as algorithm does internally)
    in_range = (df["x"] >= lo) & (df["x"] <= hi)
    df_filtered = df[in_range].copy()
    N_filtered = len(df_filtered)

    if N_filtered < 10:
        return None  # skip degenerate cases

    pdf_true_vals = true_pdf(df_filtered["x"].values, distribution=distribution)

    # Per-point edge flag and nbins
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    x_vals = df_filtered["x"].values
    bin_idx = np.clip(np.digitize(x_vals, bin_edges) - 1, 0, n_bins - 1)
    is_edge = ((bin_idx == 0) | (bin_idx == n_bins - 1)).astype(np.int8)

    # Initialize result arrays for filtered points
    smooth_pdf = np.full(N_filtered, np.nan, dtype=np.float32)
    smooth_thr = np.full(N_filtered, np.nan, dtype=np.float32)
    smooth_is = np.zeros(N_filtered, dtype=np.int8)
    v5_pdf = np.full(N_filtered, np.nan, dtype=np.float32)
    v5_thr = np.full(N_filtered, np.nan, dtype=np.float32)
    v5_is = np.zeros(N_filtered, dtype=np.int8)

    # --- Smooth legacy ---
    try:
        sampled = downsampleDFSmoothFactorized(
            df_filtered, frac=frac,
            variables={"x": (n_bins, lo, hi)},
            random_state=iteration * 100 + 1,
            debug=False,  # production mode: _pdf and _threshold always present
        )
        idx = sampled.index.values
        smooth_pdf[idx] = sampled["_pdf"].values.astype(np.float32)
        smooth_thr[idx] = sampled["_threshold"].values.astype(np.float32)
        smooth_is[idx] = 1
    except Exception as e:
        print(f"  WARNING: smooth failed iter {iteration}: {e}")

    # --- Smooth v5 ---
    try:
        sampled_v5 = downsampleDFSmoothFactorized(
            df_filtered, frac=frac,
            variables={"x": (n_bins, lo, hi)},
            random_state=iteration * 100 + 2,
            debug=False,
            pdf_params=pdf_params_v5 or PDF_PARAMS_V5,
        )
        idx = sampled_v5.index.values
        v5_pdf[idx] = sampled_v5["_pdf"].values.astype(np.float32)
        v5_thr[idx] = sampled_v5["_threshold"].values.astype(np.float32)
        v5_is[idx] = 1
    except Exception as e:
        print(f"  WARNING: smooth_v5 failed iter {iteration}: {e}")

    # Keep only sampled rows (union of both methods)
    is_any = (smooth_is == 1) | (v5_is == 1)
    if is_any.sum() == 0:
        return None

    # Build output DataFrame — sampled rows only
    result = pd.DataFrame({
        "iteration": np.int16(iteration),
        "x": df_filtered["x"].values[is_any].astype(np.float16),
        "pdf_true": pdf_true_vals[is_any].astype(np.float32),
        # Per-iteration metadata (constant per iteration)
        "N": np.int32(N),
        "frac": np.float16(frac),
        "dx": np.float16(dx),
        "isEdge": is_edge[is_any],
        "nBins": np.int16(n_bins),
        # Smooth legacy
        "smooth_pdf": smooth_pdf[is_any].astype(np.float16),
        "smooth_threshold": np.float32(np.round(smooth_thr[is_any], 5)),
        "smooth_is_sampled": smooth_is[is_any],
        # Smooth v5
        "smooth_v5_pdf": v5_pdf[is_any].astype(np.float16),
        "smooth_v5_threshold": np.float32(np.round(v5_thr[is_any], 5)),
        "smooth_v5_is_sampled": v5_is[is_any],
    })

    return result


def run_one_iteration_quantile(iteration, N, frac, nbins, pdf_params_v5=None, distribution="gaussian"):
    """Run one scan iteration with Gaussian CDF-quantile bin edges.

    Bin edges: norm.ppf(np.linspace(0.001, 0.999, nbins+1))
    Per-point dx: actual bin width at each point's position.
    With quantile bins: pdf(x) × dx ≈ 1/nbins = constant → λ ≈ N/nbins.
    """
    from scipy.stats import norm as sp_norm

    rng_data = np.random.RandomState(iteration)

    # Quantile bin edges (Gaussian quantiles — works for any distribution)
    edges = sp_norm.ppf(np.linspace(0.001, 0.999, nbins + 1)) * SIGMA
    lo, hi = edges[0], edges[-1]

    # Generate data
    x = generate_data(rng_data, N, distribution=distribution)
    df = pd.DataFrame({"x": x})

    # Filter to bin range
    in_range = (df["x"] >= lo) & (df["x"] <= hi)
    df_filtered = df[in_range].reset_index(drop=True)
    N_filtered = len(df_filtered)

    if N_filtered < 10:
        return None

    pdf_true_vals = true_pdf(df_filtered["x"].values, distribution=distribution)
    x_vals = df_filtered["x"].values

    # Per-point dx: actual bin width at each point's position
    bin_idx = np.clip(np.digitize(x_vals, edges) - 1, 0, nbins - 1)
    dx_per_point = (edges[bin_idx + 1] - edges[bin_idx]).astype(np.float32)
    is_edge = ((bin_idx == 0) | (bin_idx == nbins - 1)).astype(np.int8)
    nbins_arr = np.full(N_filtered, nbins, dtype=np.int16)

    # Initialize result arrays
    smooth_pdf = np.full(N_filtered, np.nan, dtype=np.float32)
    smooth_thr = np.full(N_filtered, np.nan, dtype=np.float32)
    smooth_is = np.zeros(N_filtered, dtype=np.int8)
    v5_pdf = np.full(N_filtered, np.nan, dtype=np.float32)
    v5_thr = np.full(N_filtered, np.nan, dtype=np.float32)
    v5_is = np.zeros(N_filtered, dtype=np.int8)

    # Option D: pass explicit edges array
    variables = {"x": edges}

    # --- Smooth legacy ---
    try:
        sampled = downsampleDFSmoothFactorized(
            df_filtered, frac=frac,
            variables=variables,
            random_state=iteration * 100 + 1,
            debug=False,
        )
        idx = sampled.index.values
        smooth_pdf[idx] = sampled["_pdf"].values.astype(np.float32)
        smooth_thr[idx] = sampled["_threshold"].values.astype(np.float32)
        smooth_is[idx] = 1
    except Exception as e:
        print(f"  WARNING: smooth failed iter {iteration}: {e}")

    # --- Smooth v5 ---
    try:
        sampled_v5 = downsampleDFSmoothFactorized(
            df_filtered, frac=frac,
            variables=variables,
            random_state=iteration * 100 + 2,
            debug=False,
            pdf_params=pdf_params_v5 or PDF_PARAMS_V5,
        )
        idx = sampled_v5.index.values
        v5_pdf[idx] = sampled_v5["_pdf"].values.astype(np.float32)
        v5_thr[idx] = sampled_v5["_threshold"].values.astype(np.float32)
        v5_is[idx] = 1
    except Exception as e:
        print(f"  WARNING: smooth_v5 failed iter {iteration}: {e}")

    # Keep only sampled rows
    is_any = (smooth_is == 1) | (v5_is == 1)
    if is_any.sum() == 0:
        return None

    result = pd.DataFrame({
        "iteration": np.int16(iteration),
        "x": x_vals[is_any].astype(np.float16),
        "pdf_true": pdf_true_vals[is_any].astype(np.float32),
        "N": np.int32(N),
        "frac": np.float16(frac),
        "dx": dx_per_point[is_any].astype(np.float16),  # per-point bin width
        "isEdge": is_edge[is_any],
        "nBins": nbins_arr[is_any],
        # Smooth legacy
        "smooth_pdf": smooth_pdf[is_any].astype(np.float16),
        "smooth_threshold": np.float32(np.round(smooth_thr[is_any], 5)),
        "smooth_is_sampled": smooth_is[is_any],
        # Smooth v5
        "smooth_v5_pdf": v5_pdf[is_any].astype(np.float16),
        "smooth_v5_threshold": np.float32(np.round(v5_thr[is_any], 5)),
        "smooth_v5_is_sampled": v5_is[is_any],
    })

    return result


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate scan tree for parameter validation")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations (default: 1000)")
    parser.add_argument("--output", type=str, default="scan_validation.root", help="Output file")
    parser.add_argument("--seed", type=int, default=42, help="Master RNG seed")
    parser.add_argument("--mode", type=str, default="uniform", choices=["uniform", "quantile"],
                       help="Binning mode: 'uniform' (fixed Δx) or 'quantile' (CDF-quantile edges)")
    parser.add_argument("--fit_coordinate", type=str, default="x", choices=["x", "bin"],
                       help="Polynomial fit coordinate: 'x' (data units) or 'bin' (bin index)")
    parser.add_argument("--nbins_min", type=int, default=20, help="Min nbins for quantile mode")
    parser.add_argument("--nbins_max", type=int, default=500, help="Max nbins for quantile mode")
    parser.add_argument("--distribution", type=str, default="gaussian",
                       choices=["gaussian", "linear"],
                       help="Data distribution: 'gaussian' or 'linear' (ramp on [-5,5])")
    args = parser.parse_args()

    n_iter = args.n_iter
    output_file = args.output
    mode = args.mode

    # Build pdf_params with fit_coordinate — store in module for iteration functions
    _pdf_params_v5 = dict(PDF_PARAMS_V5)
    _pdf_params_v5["fit_coordinate"] = args.fit_coordinate

    print("=" * 70)
    print(f"GENERATING SCAN TREE — mode={mode}, distribution={args.distribution}")
    print("=" * 70)
    print(f"\nScan parameters:")
    print(f"  N_ITERATIONS = {n_iter}")
    print(f"  N range      = [{N_MIN}, {N_MAX}] (log-uniform)")
    print(f"  frac range   = [{FRAC_MIN}, {FRAC_MAX}] (uniform)")
    if mode == "uniform":
        print(f"  Δx range     = [{DX_MIN}, {DX_MAX}] (uniform)")
    else:
        print(f"  nbins range  = [{args.nbins_min}, {args.nbins_max}] (uniform int)")
        print(f"  Edges        = norm.ppf(linspace(0.001, 0.999, nbins+1)) × σ")
        print(f"  dx           = per-point actual bin width")
    print(f"  Methods      = smooth (legacy), smooth_v5")
    print(f"  PDF_PARAMS_V5= {_pdf_params_v5}")
    print(f"  Distribution = {args.distribution}")
    print(f"  OUTPUT       = {output_file}")
    print(f"  Seed         = {args.seed}")

    # Draw all scan parameters upfront (reproducible)
    master_rng = np.random.RandomState(args.seed)
    scan_params = []
    for i in range(n_iter):
        N, frac, dx, nbins = draw_scan_params(master_rng, mode=mode,
                                               nbins_min=args.nbins_min,
                                               nbins_max=args.nbins_max)
        scan_params.append((i, N, frac, dx, nbins))

    # Print parameter summary
    Ns = [p[1] for p in scan_params]
    fracs = [p[2] for p in scan_params]
    total_points = sum(Ns)
    print(f"\nParameter summary ({n_iter} iterations):")
    print(f"  N:    [{min(Ns)}, {max(Ns)}], median={int(np.median(Ns))}")
    print(f"  frac: [{min(fracs):.4f}, {max(fracs):.4f}], median={np.median(fracs):.4f}")
    if mode == "uniform":
        dxs = [p[3] for p in scan_params]
        print(f"  Δx:   [{min(dxs):.4f}, {max(dxs):.4f}], median={np.median(dxs):.4f}")
    else:
        nbinss = [p[4] for p in scan_params]
        print(f"  nbins: [{min(nbinss)}, {max(nbinss)}], median={int(np.median(nbinss))}")
    print(f"  Total input points: {total_points:,} (~{total_points * 0.06:,.0f} sampled)")

    # Run iterations
    print(f"\nRunning {n_iter} iterations...")
    t0 = time.time()
    all_results = []
    n_failed = 0

    for i, (iteration, N, frac, dx, nbins) in enumerate(scan_params):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (n_iter - i) / rate
            print(f"  [{i}/{n_iter}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s")
        elif i % 10 == 0:
            if mode == "uniform":
                print(f"  iter {i}: N={N}, frac={frac:.4f}, dx={dx:.4f}", end="", flush=True)
            else:
                print(f"  iter {i}: N={N}, frac={frac:.4f}, nbins={nbins}", end="", flush=True)

        if mode == "quantile":
            result = run_one_iteration_quantile(iteration, N, frac, nbins,
                                               pdf_params_v5=_pdf_params_v5,
                                               distribution=args.distribution)
        else:
            result = run_one_iteration(iteration, N, frac, dx,
                                      pdf_params_v5=_pdf_params_v5,
                                      distribution=args.distribution)

        if result is not None:
            all_results.append(result)
            if i % 10 == 0:
                print(f" → {len(result)} rows")
        else:
            n_failed += 1
            if i % 10 == 0:
                print(" → SKIPPED")

    elapsed = time.time() - t0
    print(f"\nDone: {len(all_results)}/{n_iter} iterations in {elapsed:.1f}s "
          f"({n_failed} failed)")

    # Concatenate
    print("\nConcatenating...")
    scan_df = pd.concat(all_results, ignore_index=True)
    print(f"  Total rows: {len(scan_df):,}")

    nbytes = scan_df.memory_usage(deep=True).sum()
    print(f"  Memory: {nbytes / 1e6:.1f} MB")

    # Params table — dx from scan tree (mean per iteration), nbins for quantile
    mean_dx_per_iter = scan_df.groupby("iteration")["dx"].apply(
        lambda s: float(np.mean(s.astype(np.float32)))).to_dict()
    params_list = []
    for p in scan_params:
        it, N_p, frac_p, dx_p, nbins_p = p
        dx_val = mean_dx_per_iter.get(it, dx_p)
        params_list.append((it, N_p, frac_p, dx_val, nbins_p))
    params_df = pd.DataFrame(params_list,
                             columns=["iteration", "N", "frac", "dx", "nbins"])
    params_df["iteration"] = params_df["iteration"].astype(np.int16)
    params_df["N"] = params_df["N"].astype(np.int32)
    params_df["frac"] = params_df["frac"].astype(np.float32)
    params_df["dx"] = params_df["dx"].astype(np.float32)
    params_df["nbins"] = params_df["nbins"].astype(np.int32)

    # Analysis range: where validation figures should be computed
    if mode == "quantile":
        # Quantile edges define the range
        from scipy.stats import norm as sp_norm
        q_edges = sp_norm.ppf(np.linspace(0.001, 0.999, 2)) * SIGMA
        params_df["x_lo"] = np.float32(q_edges[0])
        params_df["x_hi"] = np.float32(q_edges[1])
    elif args.distribution == "linear":
        params_df["x_lo"] = np.float32(-LINEAR_HALF)
        params_df["x_hi"] = np.float32(LINEAR_HALF)
    else:
        # Gaussian uniform: ±3σ analysis range (data extends to ±6σ)
        params_df["x_lo"] = np.float32(-3 * SIGMA)
        params_df["x_hi"] = np.float32(3 * SIGMA)

    print(f"  Params table: {len(params_df)} rows")
    print(f"  Analysis range: [{params_df['x_lo'].iloc[0]:.2f}, {params_df['x_hi'].iloc[0]:.2f}]")

    # Export
    if HAS_UPROOT:
        print(f"\nExporting to {output_file}...")

        # ROOT TTrees don't support float16 or int8 — upcast for writing
        # Data is computed in float16 precision (rounded), stored as float32 in tree
        write_df = scan_df.copy()
        for col in write_df.columns:
            if write_df[col].dtype == np.float16:
                write_df[col] = write_df[col].astype(np.float32)
            elif write_df[col].dtype == np.int8:
                write_df[col] = write_df[col].astype(np.int32)
            elif write_df[col].dtype == np.int16:
                write_df[col] = write_df[col].astype(np.int32)

        scan_data = {col: write_df[col].values for col in write_df.columns}
        params_data = {col: params_df[col].values for col in params_df.columns}

        with uproot.recreate(output_file) as f:
            f["scan"] = scan_data
            f["params"] = params_data

        file_size = os.path.getsize(output_file)
        print(f"  Saved: {output_file} ({file_size / 1e6:.1f} MB)")
        print(f"  Trees: 'scan' ({len(scan_df):,} rows), 'params' ({len(params_df)} rows)")

    else:
        csv_file = output_file.replace(".root", ".csv")
        params_csv = output_file.replace(".root", "_params.csv")
        print(f"\nExporting to CSV...")
        scan_df.to_csv(csv_file, index=False)
        params_df.to_csv(params_csv, index=False)
        print(f"  Saved: {csv_file}")
        print(f"  Saved: {params_csv}")

    # Column summary
    print(f"\nColumn summary:")
    print(f"  Common:     iteration (int16), x (float16), pdf_true (float32)")
    print(f"  Metadata:   N (int32), frac (float16), dx (float16)")
    print(f"  smooth:     smooth_pdf (float16), smooth_threshold (float32), smooth_is_sampled (int8)")
    print(f"  smooth_v5:  smooth_v5_pdf (float16), smooth_v5_threshold (float32), smooth_v5_is_sampled (int8)")
    print(f"  Weights derivable: w = 1/max(pdf, threshold)")

    print(f"\n{'='*70}")
    print(f"SCAN COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
