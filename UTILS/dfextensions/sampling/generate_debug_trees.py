#!/usr/bin/env python
"""
generate_debug_trees.py

Generate debug trees for downsampling validation.

Runs 50 iterations of each downsampling method with different parameters.
Exports two ROOT trees:
  - 'full':    ALL rows from all iterations (50 × N = 5M rows)
  - 'sampled': Only selected rows (~50 × N × frac = 500k rows)

Column naming convention:
  {method}_D{binwidth}_R{range}_F{frac}_{quantity}
  
  Methods:
    - binned:   downsampleDF (pre-binned stratify)
    - smooth:   downsampleDFSmoothFactorized
    - smoothND: downsampleDFSmooth (full ND, for comparison)
  
  Parameters:
    - D01, D02, D05: bin width = 0.1, 0.2, 0.5
    - R6: range = ±6 (for σ=1 Gaussian, this is ±6σ)
    - F01: frac = 0.1
  
  Quantities:
    - pdf:        empirical PDF at this point (_debug_pdf from algorithm)
    - weight_raw: 1/PDF before normalization (_debug_weight_raw)
    - weight:     stored weight (normalized sampling probability)
    - is_sampled: 1 if selected, 0 otherwise

Common columns:
    - iteration:  0-49
    - x:          variable value
    - pdf_true:   analytical Gaussian PDF

Usage:
    python generate_debug_trees.py
    → creates debug_sampling.root

Inspect with:
    root -l debug_sampling.root
    full->Draw("smooth_D01_R6_F01_pdf / pdf_true : x")
    sampled->Draw("smooth_D01_R6_F01_weight : x")
"""

import numpy as np
import pandas as pd
import sys
import os

# Import from the downsample module
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from dfextensions.sampling.downsample import (
    downsampleDF,
    downsampleDFSmoothFactorized,
    downsampleDFSmooth,
)

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
    print("WARNING: uproot not available, will save as CSV instead")


# ===================================================================
# Configuration - HIGH STATISTICS VERSION
# ===================================================================

N_ITERATIONS = 100     # 100 iterations for variance estimation
N_POINTS = 100_000     # 100k points per iteration
SIGMA = 1.0
FRAC = 0.1
RANGE = 6  # ±6 (in units of sigma, so ±6σ)

# Bin widths to test (Δx ≤ 0.3, consistent with v5 validated range)
BIN_WIDTHS = [0.05, 0.1, 0.2]

# PDF estimator configurations
# Legacy: no pdf_params (log-interp + linear interp)
# v5: 3-layer (kernel + Poisson + parabolic), validated winner from fig1-fig3
PDF_PARAMS_V5 = {"poly_order": 2, "poly_half_range": 0.5, "kernel_sigma_bins": 0.5}

# Output file
OUTPUT_FILE = "validation_sampling.root"


# ===================================================================
# Helper functions
# ===================================================================

def gaussian_pdf(x, mu=0, sigma=SIGMA):
    """Analytical Gaussian PDF."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def make_gaussian_data(n=N_POINTS, sigma=SIGMA, seed=0):
    """Generate Gaussian data."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({"x": rng.normal(0, sigma, n)})


def run_one_iteration(iteration: int) -> pd.DataFrame:
    """
    Run one iteration: generate data, run all methods, collect results.
    
    Returns a DataFrame with all rows and all columns.
    """
    print(f"  Iteration {iteration}...", end=" ", flush=True)
    
    # Generate data (different seed each iteration)
    df = make_gaussian_data(n=N_POINTS, sigma=SIGMA, seed=iteration)
    N = len(df)
    
    # True PDF
    pdf_true = gaussian_pdf(df["x"].values)
    
    # Initialize result DataFrame
    result = pd.DataFrame({
        "iteration": np.full(N, iteration, dtype=np.int32),
        "x": df["x"].values,
        "pdf_true": pdf_true,
    })
    
    # Pre-bin for binned method
    for bw in BIN_WIDTHS:
        bin_col = f"x_bin_{int(bw*100):02d}"
        df[bin_col] = (np.floor(df["x"] / bw) * bw).astype(np.float32)
    
    # Helper function to replace NaN with -1
    def fill_nan(arr, fill_value=-1.0):
        arr = arr.copy()
        arr[np.isnan(arr)] = fill_value
        return arr
    
    # ---------------------------------------------------------------
    # Run each method with each bin width
    # ---------------------------------------------------------------
    
    for bw in BIN_WIDTHS:
        bw_str = f"D{int(bw*100):02d}"  # D01, D02, D05
        r_str = f"R{RANGE}"              # R6
        f_str = f"F{int(FRAC*100):02d}"  # F01
        
        n_bins = int(2 * RANGE / bw)
        lo, hi = -RANGE, RANGE
        
        # --- BINNED method ---
        prefix = f"binned_{bw_str}_{r_str}_{f_str}"
        bin_col = f"x_bin_{int(bw*100):02d}"
        
        try:
            sampled_binned = downsampleDF(
                df, frac=FRAC, stratify=bin_col,
                random_state=iteration * 1000,
                debug=True,
            )
            
            # Mark which rows were sampled
            is_sampled_binned = np.zeros(N, dtype=np.int32)
            is_sampled_binned[sampled_binned.index] = 1
            
            # Get debug columns for sampled rows, fill NaN for others
            pdf_col = np.full(N, np.nan)
            weight_raw_col = np.full(N, np.nan)
            weight_col = np.full(N, np.nan)
            
            pdf_col[sampled_binned.index] = sampled_binned["_debug_pdf"].values
            weight_raw_col[sampled_binned.index] = sampled_binned["_debug_weight_raw"].values
            weight_col[sampled_binned.index] = sampled_binned["weight"].values
            
            result[f"{prefix}_pdf"] = fill_nan(pdf_col)
            result[f"{prefix}_weight_raw"] = fill_nan(weight_raw_col)
            result[f"{prefix}_weight"] = fill_nan(weight_col)
            result[f"{prefix}_is_sampled"] = is_sampled_binned
            
        except Exception as e:
            print(f"WARNING: binned {bw_str} failed: {e}")
            result[f"{prefix}_pdf"] = -1.0
            result[f"{prefix}_weight_raw"] = -1.0
            result[f"{prefix}_weight"] = -1.0
            result[f"{prefix}_is_sampled"] = 0
        
        # --- SMOOTH FACTORIZED method ---
        prefix = f"smooth_{bw_str}_{r_str}_{f_str}"
        
        try:
            # Filter to range first (same as algorithm does internally)
            in_range = (df["x"] >= lo) & (df["x"] <= hi)
            df_filtered = df[in_range].copy()
            original_indices = df_filtered.index.values
            
            sampled_smooth = downsampleDFSmoothFactorized(
                df_filtered, frac=FRAC,
                variables={"x": (n_bins, lo, hi)},
                random_state=iteration * 1000 + 1,
                debug=True,
            )
            
            # Initialize columns
            pdf_col = np.full(N, np.nan, dtype=np.float32)
            weight_raw_col = np.full(N, np.nan, dtype=np.float32)
            weight_col = np.full(N, np.nan, dtype=np.float32)
            is_sampled_smooth = np.zeros(N, dtype=np.int32)
            
            # The sampled df has reset index 0..n_sampled-1
            # Map back using the filtered df's original indices
            sampled_original_idx = original_indices[sampled_smooth.index.values]
            
            pdf_col[sampled_original_idx] = sampled_smooth["_debug_pdf"].values
            weight_raw_col[sampled_original_idx] = sampled_smooth["_debug_weight_raw"].values
            weight_col[sampled_original_idx] = sampled_smooth["weight"].values
            is_sampled_smooth[sampled_original_idx] = 1
            
            # Store threshold (needed for HT reconstruction weight)
            threshold_val = sampled_smooth.attrs.get("threshold", np.nan)
            
            result[f"{prefix}_pdf"] = fill_nan(pdf_col)
            result[f"{prefix}_weight_raw"] = fill_nan(weight_raw_col)
            result[f"{prefix}_weight"] = fill_nan(weight_col)
            result[f"{prefix}_is_sampled"] = is_sampled_smooth
            result[f"{prefix}_threshold"] = np.float32(threshold_val)
            
        except Exception as e:
            print(f"WARNING: smooth {bw_str} failed: {e}")
            result[f"{prefix}_pdf"] = -1.0
            result[f"{prefix}_weight_raw"] = -1.0
            result[f"{prefix}_weight"] = -1.0
            result[f"{prefix}_is_sampled"] = 0
            result[f"{prefix}_threshold"] = np.float32(np.nan)
        
        # --- SMOOTH ND method (1D case = should be same as factorized) ---
        prefix = f"smoothND_{bw_str}_{r_str}_{f_str}"
        
        try:
            # Use same filtered df
            sampled_nd = downsampleDFSmooth(
                df_filtered, frac=FRAC,
                variables={"x": (n_bins, lo, hi)},
                random_state=iteration * 1000 + 2,
                debug=True,
            )
            
            pdf_col = np.full(N, np.nan, dtype=np.float32)
            weight_raw_col = np.full(N, np.nan, dtype=np.float32)
            weight_col = np.full(N, np.nan, dtype=np.float32)
            is_sampled_nd = np.zeros(N, dtype=np.int32)
            
            sampled_original_idx = original_indices[sampled_nd.index.values]
            
            pdf_col[sampled_original_idx] = sampled_nd["_debug_pdf"].values
            weight_raw_col[sampled_original_idx] = sampled_nd["_debug_weight_raw"].values
            weight_col[sampled_original_idx] = sampled_nd["weight"].values
            is_sampled_nd[sampled_original_idx] = 1
            
            threshold_val = sampled_nd.attrs.get("threshold", np.nan)
            
            result[f"{prefix}_pdf"] = fill_nan(pdf_col)
            result[f"{prefix}_weight_raw"] = fill_nan(weight_raw_col)
            result[f"{prefix}_weight"] = fill_nan(weight_col)
            result[f"{prefix}_is_sampled"] = is_sampled_nd
            result[f"{prefix}_threshold"] = np.float32(threshold_val)
            
        except Exception as e:
            print(f"WARNING: smoothND {bw_str} failed: {e}")
            result[f"{prefix}_pdf"] = -1.0
            result[f"{prefix}_weight_raw"] = -1.0
            result[f"{prefix}_weight"] = -1.0
            result[f"{prefix}_is_sampled"] = 0
            result[f"{prefix}_threshold"] = np.float32(np.nan)
        
        # --- SMOOTH v5 method (3-layer: kernel+Poisson+parabolic) ---
        prefix = f"smooth_v5_{bw_str}_{r_str}_{f_str}"
        
        try:
            sampled_v5 = downsampleDFSmoothFactorized(
                df_filtered, frac=FRAC,
                variables={"x": (n_bins, lo, hi)},
                random_state=iteration * 1000 + 3,
                debug=True,
                pdf_params=PDF_PARAMS_V5,
            )
            
            pdf_col = np.full(N, np.nan, dtype=np.float32)
            weight_raw_col = np.full(N, np.nan, dtype=np.float32)
            weight_col = np.full(N, np.nan, dtype=np.float32)
            is_sampled_v5 = np.zeros(N, dtype=np.int32)
            
            sampled_original_idx = original_indices[sampled_v5.index.values]
            
            pdf_col[sampled_original_idx] = sampled_v5["_debug_pdf"].values
            weight_raw_col[sampled_original_idx] = sampled_v5["_debug_weight_raw"].values
            weight_col[sampled_original_idx] = sampled_v5["weight"].values
            is_sampled_v5[sampled_original_idx] = 1
            
            threshold_val = sampled_v5.attrs.get("threshold", np.nan)
            
            result[f"{prefix}_pdf"] = fill_nan(pdf_col)
            result[f"{prefix}_weight_raw"] = fill_nan(weight_raw_col)
            result[f"{prefix}_weight"] = fill_nan(weight_col)
            result[f"{prefix}_is_sampled"] = is_sampled_v5
            result[f"{prefix}_threshold"] = np.float32(threshold_val)
            
        except Exception as e:
            print(f"WARNING: smooth_v5 {bw_str} failed: {e}")
            result[f"{prefix}_pdf"] = -1.0
            result[f"{prefix}_weight_raw"] = -1.0
            result[f"{prefix}_weight"] = -1.0
            result[f"{prefix}_is_sampled"] = 0
            result[f"{prefix}_threshold"] = np.float32(np.nan)
        
        # --- SMOOTH ND v5 method ---
        prefix = f"smoothND_v5_{bw_str}_{r_str}_{f_str}"
        
        try:
            sampled_ndv5 = downsampleDFSmooth(
                df_filtered, frac=FRAC,
                variables={"x": (n_bins, lo, hi)},
                random_state=iteration * 1000 + 4,
                debug=True,
                pdf_params=PDF_PARAMS_V5,
            )
            
            pdf_col = np.full(N, np.nan, dtype=np.float32)
            weight_raw_col = np.full(N, np.nan, dtype=np.float32)
            weight_col = np.full(N, np.nan, dtype=np.float32)
            is_sampled_ndv5 = np.zeros(N, dtype=np.int32)
            
            sampled_original_idx = original_indices[sampled_ndv5.index.values]
            
            pdf_col[sampled_original_idx] = sampled_ndv5["_debug_pdf"].values
            weight_raw_col[sampled_original_idx] = sampled_ndv5["_debug_weight_raw"].values
            weight_col[sampled_original_idx] = sampled_ndv5["weight"].values
            is_sampled_ndv5[sampled_original_idx] = 1
            
            threshold_val = sampled_ndv5.attrs.get("threshold", np.nan)
            
            result[f"{prefix}_pdf"] = fill_nan(pdf_col)
            result[f"{prefix}_weight_raw"] = fill_nan(weight_raw_col)
            result[f"{prefix}_weight"] = fill_nan(weight_col)
            result[f"{prefix}_is_sampled"] = is_sampled_ndv5
            result[f"{prefix}_threshold"] = np.float32(threshold_val)
            
        except Exception as e:
            print(f"WARNING: smoothND_v5 {bw_str} failed: {e}")
            result[f"{prefix}_pdf"] = -1.0
            result[f"{prefix}_weight_raw"] = -1.0
            result[f"{prefix}_weight"] = -1.0
            result[f"{prefix}_is_sampled"] = 0
            result[f"{prefix}_threshold"] = np.float32(np.nan)
    
    print("done")
    return result


def main():
    print("="*70)
    print("GENERATING DEBUG TREES FOR DOWNSAMPLING VALIDATION")
    print("="*70)
    print(f"\nParameters:")
    print(f"  N_ITERATIONS = {N_ITERATIONS}")
    print(f"  N_POINTS     = {N_POINTS}")
    print(f"  SIGMA        = {SIGMA}")
    print(f"  FRAC         = {FRAC}")
    print(f"  RANGE        = ±{RANGE}")
    print(f"  BIN_WIDTHS   = {BIN_WIDTHS}")
    print(f"  METHODS      = binned, smooth, smoothND, smooth_v5, smoothND_v5")
    print(f"  PDF_PARAMS_V5= {PDF_PARAMS_V5}")
    print(f"  OUTPUT       = {OUTPUT_FILE}")
    
    # Run all iterations
    print(f"\nRunning {N_ITERATIONS} iterations...")
    all_results = []
    for i in range(N_ITERATIONS):
        result = run_one_iteration(i)
        all_results.append(result)
    
    # Concatenate all results
    print("\nConcatenating results...")
    full_df = pd.concat(all_results, ignore_index=True)
    print(f"  Total rows: {len(full_df)}")
    
    # Create sampled-only DataFrame
    # A row is "sampled" if it was selected by ANY method
    is_sampled_cols = [c for c in full_df.columns if c.endswith("_is_sampled")]
    is_sampled_any = full_df[is_sampled_cols].max(axis=1) > 0
    sampled_df = full_df[is_sampled_any].copy()
    print(f"  Sampled rows: {len(sampled_df)}")
    
    # Export
    if HAS_UPROOT:
        print(f"\nExporting to {OUTPUT_FILE}...")
        
        # Convert to dict of arrays for uproot
        full_data = {col: full_df[col].values for col in full_df.columns}
        sampled_data = {col: sampled_df[col].values for col in sampled_df.columns}
        
        with uproot.recreate(OUTPUT_FILE) as f:
            f["full"] = full_data
            f["sampled"] = sampled_data
        
        print(f"  Saved: {OUTPUT_FILE}")
        print(f"  Trees: 'full' ({len(full_df)} rows), 'sampled' ({len(sampled_df)} rows)")
        
    else:
        # Fallback to CSV
        full_csv = OUTPUT_FILE.replace(".root", "_full.csv")
        sampled_csv = OUTPUT_FILE.replace(".root", "_sampled.csv")
        
        print(f"\nExporting to CSV (uproot not available)...")
        full_df.to_csv(full_csv, index=False)
        sampled_df.to_csv(sampled_csv, index=False)
        print(f"  Saved: {full_csv} ({len(full_df)} rows)")
        print(f"  Saved: {sampled_csv} ({len(sampled_df)} rows)")
    
    # Print column summary
    print(f"\nColumns in output:")
    print(f"  Common: iteration, x, pdf_true")
    print(f"  Per method/binwidth:")
    for bw in BIN_WIDTHS:
        bw_str = f"D{int(bw*100):02d}"
        print(f"    *_{bw_str}_R{RANGE}_F{int(FRAC*100):02d}_[pdf|weight_raw|weight|is_sampled|threshold]")
    print(f"  Methods: binned, smooth, smoothND, smooth_v5, smoothND_v5")
    print(f"  (threshold column only for smooth* methods)")
    
    print(f"\n" + "=" * 70)
    print(f"EXPECTED RUNTIME: ~30 seconds for 100 iterations × 100k points")
    print(f"OUTPUT SIZE: ~500MB")
    print(f"=" * 70)
    
    # Print verification hints
    print(f"\n" + "="*70)
    print("VERIFICATION COMMANDS")
    print("="*70)
    print("""
# In ROOT:
root -l debug_sampling.root

# Check PDF estimation accuracy (should be ~1.0 with some scatter)
full->Draw("smooth_D01_R6_F01_pdf / pdf_true : x", "smooth_D01_R6_F01_is_sampled")

# Check weight consistency (pdf × weight_raw should be constant)
sampled->Draw("smooth_D01_R6_F01_pdf * smooth_D01_R6_F01_weight_raw : x")

# Compare methods
sampled->Draw("smooth_D01_R6_F01_pdf : x", "", "")
sampled->Draw("binned_D01_R6_F01_pdf : x", "", "same")

# Statistical properties across iterations
sampled->Draw("smooth_D01_R6_F01_weight >> h(100,0,0.001)", "iteration<50")

# In Python with uproot:
import uproot
import matplotlib.pyplot as plt

f = uproot.open("debug_sampling.root")
full = f["full"].arrays(library="pd")
sampled = f["sampled"].arrays(library="pd")

# Check ratio of empirical to true PDF
mask = sampled["smooth_D01_R6_F01_is_sampled"] == 1
plt.scatter(sampled.loc[mask, "x"], 
            sampled.loc[mask, "smooth_D01_R6_F01_pdf"] / sampled.loc[mask, "pdf_true"],
            s=1, alpha=0.1)
plt.axhline(1.0, color='r')
plt.xlabel("x")
plt.ylabel("empirical PDF / true PDF")
plt.show()
""")


if __name__ == "__main__":
    main()
