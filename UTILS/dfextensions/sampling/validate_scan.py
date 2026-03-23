#!/usr/bin/env python
"""
validate_scan.py

Validate theoretical scaling laws using the parameter scan tree.

Primary figures (physics observables):
  S1: asinh(N_sampled/(N*frac) - 1) vs frac     — bisection accuracy
  S2: σ(threshold residuals) vs 1/√(N×frac)     — threshold stability
  S3: RMS(pdf_emp/pdf_true - 1) vs 1/√λ         — PDF estimator bias
  S4a: reweighted_hist / expected_hist by frac       — spectra recovery (per x-bin)
  S4b: same by Δx, color N_sampled                   — spectra recovery (per x-bin)
  S4c: σ_meas vs σ_model scatter + pull              — σ scaling validation

Usage:
    python validate_scan.py --input scan_test1000.root --output figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats
from scipy.optimize import curve_fit
import os
import argparse
import time


# ===================================================================
# Configuration
# ===================================================================

METHOD_COLORS = {"smooth": "royalblue", "smooth_v5": "crimson"}
METHOD_LABELS = {"smooth": "Legacy (log-interp)", "smooth_v5": "v5 (kernel+Poisson+parabolic)"}
SIGMA = 1.0
N_COLORS_3 = ["#1b9e77", "#d95f02", "#7570b3"]
PDF_BINS = [(0.01, 0.05), (0.05, 0.15), (0.15, 0.40)]
PDF_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]
PDF_LABELS = [f"pdf:[{lo},{hi}]" for lo, hi in PDF_BINS]

_report_lines = []
_all_figures = []


def report(text):
    _report_lines.append(text)
    print(text)


def report_table(headers, rows):
    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
              for i, h in enumerate(headers)]
    hdr = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    report(hdr)
    report(sep)
    for row in rows:
        report("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")


def gaussian_pdf(x, sigma=SIGMA):
    return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def derive_scan_metadata(scan, params=None, x_range=None):
    """Derive distribution-agnostic metadata from scan tree.

    Priority for x_range: (1) x_range arg (from --x_range CLI),
    (2) params table x_lo/x_hi, (3) data min/max.
    """
    x_all = scan["x"].values.astype(np.float64)
    pdf_all = scan["pdf_true"].values.astype(np.float64)

    # Determine x range: CLI > params table > data
    if x_range is not None:
        x_lo, x_hi = float(x_range[0]), float(x_range[1])
    elif params is not None and "x_lo" in params.columns:
        x_lo = float(params["x_lo"].iloc[0])
        x_hi = float(params["x_hi"].iloc[0])
    else:
        x_lo = float(np.min(x_all))
        x_hi = float(np.max(x_all))
    x_mid = 0.5 * (x_lo + x_hi)
    x_full = x_hi - x_lo

    # Boundaries: split range into 3 equal parts (core/shoulder/tail)
    b1 = x_full / 6.0  # core: |x - x_mid| < b1
    b2 = x_full / 3.0  # shoulder: b1 < |x - x_mid| < b2

    # pdf_true profile: median per x-bin across all iterations
    n_profile = 200
    profile_edges = np.linspace(x_lo, x_hi, n_profile + 1)
    profile_centers = 0.5 * (profile_edges[:-1] + profile_edges[1:])
    profile_pdf = np.zeros(n_profile)
    idx = np.clip(np.digitize(x_all, profile_edges) - 1, 0, n_profile - 1)
    for i in range(n_profile):
        sel = idx == i
        if sel.sum() > 10:
            profile_pdf[i] = np.median(pdf_all[sel])

    return {
        "x_lo": x_lo, "x_hi": x_hi, "x_mid": x_mid, "x_full": x_full,
        "b1": b1, "b2": b2,  # core/shoulder/tail boundaries
        "profile_centers": profile_centers,
        "profile_pdf": profile_pdf,
    }


def pdf_true_at(x, meta):
    """Interpolate pdf_true profile at arbitrary x positions."""
    return np.interp(x, meta["profile_centers"], meta["profile_pdf"])


def expected_counts_from_profile(N, hist_bins, meta):
    """Expected histogram counts using pdf_true profile. Distribution-agnostic."""
    bin_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    bin_widths = np.diff(hist_bins)
    pdf_vals = pdf_true_at(bin_centers, meta)
    return N * pdf_vals * bin_widths


# ===================================================================
# I/O
# ===================================================================

def load_scan(input_file):
    if input_file.endswith(".root"):
        import uproot
        f = uproot.open(input_file)
        scan = f["scan"].arrays(library="pd")
        params = f["params"].arrays(library="pd")
    else:
        scan = pd.read_csv(input_file)
        params = pd.read_csv(input_file.replace(".csv", "_params.csv"))
    report(f"Loaded: {len(scan):,} scan rows, {len(params)} iterations")
    return scan, params


def savefig(fig, path, title=None):
    fig.tight_layout()
    fig.savefig(path + ".png", dpi=150, bbox_inches="tight")
    _all_figures.append((fig, title or os.path.basename(path)))
    report(f"  Saved: {path}.png")


def save_combined_pdf(output_dir):
    pdf_path = os.path.join(output_dir, "scan_report.pdf")
    with PdfPages(pdf_path) as pdf:
        for fig, title in _all_figures:
            pdf.savefig(fig, bbox_inches="tight")
    for fig, _ in _all_figures:
        plt.close(fig)
    report(f"  Saved combined: {pdf_path} ({len(_all_figures)} pages)")


# ===================================================================
# Helpers
# ===================================================================

def add_logN_bins(d, n_bins=3):
    d["logN"] = np.log10(d["N"].astype(float))
    edges = np.quantile(d["logN"], np.linspace(0, 1, n_bins + 1))
    edges[0] -= 0.01; edges[-1] += 0.01
    labels = [f"N:[{10**edges[i]:.0f},{10**edges[i+1]:.0f}]" for i in range(n_bins)]
    d["N_bin"] = pd.cut(d["logN"], bins=edges, labels=labels, include_lowest=True)
    return labels


def add_frac_bins(d, n_bins=5):
    d["frac_bin"] = pd.qcut(d["frac"], n_bins, duplicates="drop")


def grouped_profile(sub, x_col, y_col, x_bin_col, min_per_bin=5, stat="mean"):
    grouped = sub.groupby(x_bin_col, observed=True)
    x_c, y_c, y_e = [], [], []
    for name, grp in grouped:
        valid = grp[y_col].dropna()
        if len(valid) < min_per_bin:
            continue
        x_c.append(grp[x_col].mean())
        if stat == "mean":
            y_c.append(valid.mean())
            y_e.append(valid.std() / np.sqrt(len(valid)))
        else:
            y_c.append(valid.median())
            y_e.append(valid.std())
    return np.array(x_c), np.array(y_c), np.array(y_e)


# ===================================================================
# Per-iteration observables
# ===================================================================

def compute_iteration_observables(scan, params, methods, meta=None):
    rows = []
    nan_counts = {m: {"total": 0, "nan": 0} for m in methods}
    for _, prow in params.iterrows():
        iteration = int(prow["iteration"])
        N = int(prow["N"])
        frac = float(prow["frac"])
        iter_mask = scan["iteration"] == iteration

        for method in methods:
            mask = iter_mask & (scan[f"{method}_is_sampled"] == 1)
            data = scan[mask]
            if len(data) == 0:
                continue

            pdf = data[f"{method}_pdf"].values.astype(np.float64)
            thr = data[f"{method}_threshold"].values.astype(np.float64)
            pdf_true = data["pdf_true"].values.astype(np.float64)
            x = data["x"].values.astype(np.float64)
            dx_arr = data["dx"].values.astype(np.float64)  # per-point dx
            dx_mean = float(np.mean(dx_arr))  # for grouping labels

            # Count NaN before filtering
            n_total = len(pdf)
            finite = np.isfinite(pdf) & np.isfinite(thr) & (pdf > 0)
            n_nan = n_total - finite.sum()
            nan_counts[method]["total"] += n_total
            nan_counts[method]["nan"] += n_nan

            if finite.sum() < 10:
                continue
            pdf = pdf[finite]; thr = thr[finite]; pdf_true = pdf_true[finite]
            x = x[finite]; dx_arr = dx_arr[finite]

            threshold = np.median(thr)
            N_sampled = len(pdf)

            w_cal = 1.0 / np.maximum(pdf, threshold)
            w_ht = np.maximum(1.0, pdf / threshold)
            sum_cw_over_N = w_ht.sum() / N
            N_eff = w_cal.sum() ** 2 / (w_cal ** 2).sum()

            # PDF bias in core (central 1/3 of x range)
            core_half = meta["b1"] if meta else 2 * SIGMA
            x_mid = meta["x_mid"] if meta else 0.0
            core = (np.abs(x - x_mid) < core_half) & (pdf_true > 0.01)
            if core.sum() > 10:
                bias_core = np.mean(pdf[core] / pdf_true[core] - 1.0)
                rms_bias = np.sqrt(np.mean((pdf[core] / pdf_true[core] - 1.0) ** 2))
            else:
                bias_core = np.nan
                rms_bias = np.nan

            # Per-pdf-bin RMS bias (for S3)
            pdf_bin_rms = {}
            has_edge = "isEdge" in data.columns
            edge_arr = data["isEdge"].values.astype(np.int8) if has_edge else None
            not_edge = (edge_arr == 0) if has_edge else np.ones(len(pdf), dtype=bool)
            # Apply finite filter to not_edge
            not_edge_f = not_edge[finite]

            for plo, phi in PDF_BINS:
                sel = (pdf_true >= plo) & (pdf_true < phi)
                label = f"rms_pdf_{plo:.2f}_{phi:.2f}"
                lambda_label = f"lambda_{plo:.2f}_{phi:.2f}"
                if sel.sum() > 10:
                    pdf_bin_rms[label] = np.sqrt(np.mean((pdf[sel] / pdf_true[sel] - 1.0) ** 2))
                    # λ per point = N × pdf_true_i × dx_i, then average
                    pdf_bin_rms[lambda_label] = np.mean(N * pdf_true[sel] * dx_arr[sel])
                else:
                    pdf_bin_rms[label] = np.nan
                    pdf_bin_rms[lambda_label] = np.nan

                # Edge-excluded variant
                sel_ne = sel & not_edge_f
                label_ne = f"rms_pdf_noedge_{plo:.2f}_{phi:.2f}"
                lambda_ne = f"lambda_noedge_{plo:.2f}_{phi:.2f}"
                if sel_ne.sum() > 10:
                    pdf_bin_rms[label_ne] = np.sqrt(np.mean((pdf[sel_ne] / pdf_true[sel_ne] - 1.0) ** 2))
                    pdf_bin_rms[lambda_ne] = np.mean(N * pdf_true[sel_ne] * dx_arr[sel_ne])
                else:
                    pdf_bin_rms[label_ne] = np.nan
                    pdf_bin_rms[lambda_ne] = np.nan

            # Spectra recovery (central 2/3 of x range)
            spec_half = meta["b2"] if meta else 2 * SIGMA
            hist_bins = np.linspace(x_mid - spec_half, x_mid + spec_half, 21)
            exp_counts = expected_counts_from_profile(N, hist_bins, meta) if meta else np.ones(20)
            reco, _ = np.histogram(x, bins=hist_bins, weights=w_ht)
            good = exp_counts > 10
            if good.sum() > 5:
                ratio_bins = reco[good] / exp_counts[good]
                spectra_mean = ratio_bins.mean()
                spectra_std = ratio_bins.std()
            else:
                spectra_mean = np.nan
                spectra_std = np.nan

            row = {
                "iteration": iteration, "method": method,
                "N": N, "frac": frac, "dx": dx_mean,
                "frac_sampled": N_sampled / N,
                "N_sampled": N_sampled, "N_eff": N_eff,
                "neff_over_nsamp": N_eff / N_sampled,
                "sum_cw_over_N": sum_cw_over_N,
                "threshold": threshold,
                "bias_core": bias_core, "rms_bias_core": rms_bias,
                "spectra_ratio_mean": spectra_mean,
                "spectra_ratio_std": spectra_std,
            }
            row.update(pdf_bin_rms)
            rows.append(row)

    obs = pd.DataFrame(rows)

    # Report NaN statistics
    for method in methods:
        nc = nan_counts[method]
        if nc["total"] > 0:
            pct = 100 * nc["nan"] / nc["total"]
            report(f"  {method}: {nc['nan']}/{nc['total']} NaN pdf points ({pct:.2f}%)")
            if pct > 5:
                report(f"  WARNING: {method} has >5% NaN — check generation")

    return obs


# ===================================================================
# S1: Sampling Fraction Accuracy
# ===================================================================

def figure_s1(obs, methods, output_dir):
    """S1: asinh(N_sampled/(N*frac) - 1) vs frac.
    Centered at 0. Expected width ~ 1/√(N*frac).
    """
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1: axes = [axes]
    fig.suptitle("S1: Sampling Fraction Accuracy\n"
                "y = asinh(N_sampled/(N×frac) − 1) — centered at 0 when perfect\n"
                "Expected width: σ ≈ 1/√(N×frac)", fontsize=10)

    for ax, method in zip(axes, methods):
        d = obs[obs["method"] == method].copy()
        if len(d) == 0: continue
        d["y"] = np.arcsinh(d["frac_sampled"] / d["frac"] - 1.0)
        n_labels = add_logN_bins(d, 3)
        add_frac_bins(d, 5)

        ax.scatter(d["frac"], d["y"], s=2, alpha=0.08, color="gray")

        for n_label, n_color in zip(n_labels, N_COLORS_3):
            sub = d[d["N_bin"] == n_label]
            x_c, y_med, y_sig = grouped_profile(sub, "frac", "y", "frac_bin", stat="median")
            if len(x_c) > 0:
                ax.plot(x_c, y_med, 'o-', ms=5, color=n_color, label=n_label, alpha=0.8)
                ax.fill_between(x_c, y_med - y_sig, y_med + y_sig, alpha=0.15, color=n_color)

                # Theory width: σ ≈ 1/√(N_median × frac)
                N_med = sub["N"].median()
                theory_sig = 1.0 / np.sqrt(N_med * x_c)
                ax.plot(x_c, theory_sig, '--', color=n_color, alpha=0.5, lw=1)
                ax.plot(x_c, -theory_sig, '--', color=n_color, alpha=0.5, lw=1)

        ax.axhline(0.0, color='red', ls='--', lw=1.5, label="perfect")
        ax.set_xlabel("frac (requested)")
        ax.set_ylabel("asinh(N_sampled/(N×frac) − 1)")
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10)
        ax.legend(fontsize=7)

        # Report observed vs expected width
        mean_y = d["y"].mean()
        std_y = d["y"].std()
        report(f"  S1 {method}: ⟨y⟩={mean_y:.4f}, σ(y)={std_y:.4f}")

    savefig(fig, os.path.join(output_dir, "s1_frac_accuracy"), "S1: Fraction Accuracy")


# ===================================================================
# S2: Threshold Stability
# ===================================================================

def figure_s2(obs, methods, output_dir, scan=None):
    """S2: Threshold stability via local linear fit.

    2 rows (methods) × 3 cols (Δx bins). Shared axes.
    Within each (N_bin × frac_bin) cell: fit thr = a + b*frac + c/√N.
    σ = std(residuals). Plot vs 1/√(N×frac).
    Smaller Δx → less PDF bias → smaller offset.

    If scan has nBins column, produces second figure S2 (high nBins only).
    """
    has_nbins = scan is not None and "nBins" in scan.columns

    # Get nBins per iteration if available
    if has_nbins:
        nbins_per_iter = scan.groupby("iteration")["nBins"].first()
        obs_with_nb = obs.copy()
        obs_with_nb["nBins"] = obs_with_nb["iteration"].map(nbins_per_iter)
        nbins_median = obs_with_nb["nBins"].median()
    else:
        obs_with_nb = obs.copy()
        obs_with_nb["nBins"] = 0
        nbins_median = 0

    subsets = [("all", "All iterations", obs)]
    if has_nbins and nbins_median > 0:
        obs_hi = obs_with_nb[obs_with_nb["nBins"] > nbins_median].copy()
        if len(obs_hi) > 30:
            subsets.append(("hi_nbins", f"nBins>{nbins_median:.0f}", obs_hi))

    for subset_key, subset_label, obs_sub in subsets:
        _plot_s2(obs_sub, methods, output_dir, subset_key, subset_label)


def _plot_s2(obs, methods, output_dir, subset_key, subset_label):
    """Internal S2 plot for one subset."""
    n_dx_bins = 3
    d_all = obs.copy()
    d_all["dx_bin"] = pd.qcut(d_all["dx"], n_dx_bins, duplicates="drop")
    dx_groups = sorted(d_all["dx_bin"].dropna().unique())
    n_panels = len(dx_groups)
    n_methods = len(methods)

    if n_panels == 0:
        report("  S2: no dx groups — skipping")
        return

    fig, axes = plt.subplots(n_methods, n_panels, figsize=(5 * n_panels, 5 * n_methods),
                             sharex=True, sharey=True)
    if n_methods == 1: axes = axes.reshape(1, -1)
    if n_panels == 1: axes = axes.reshape(-1, 1)

    fig.suptitle(f"S2: Threshold Stability — {subset_label}\n"
                "σ(thr − local_fit) vs 1/√(N×frac)\n"
                "Columns = Δx bins — smaller Δx → less PDF bias → smaller offset",
                fontsize=10)

    fit_table_rows = []
    global_xmax = 0
    global_ymax = 0

    for row_idx, method in enumerate(methods):
        d = d_all[d_all["method"] == method].copy()
        if len(d) == 0: continue

        d["inv_sqrt_N"] = 1.0 / np.sqrt(d["N"].astype(float))
        d["inv_sqrt_Nf"] = 1.0 / np.sqrt(d["N"].astype(float) * d["frac"])

        for col_idx, dx_grp in enumerate(dx_groups):
            ax = axes[row_idx, col_idx]
            sub = d[d["dx_bin"] == dx_grp].copy()
            dx_mean = sub["dx"].mean()

            if len(sub) < 30:
                ax.text(0.5, 0.5, f"Too few ({len(sub)})",
                       transform=ax.transAxes, ha='center', va='center')
                continue

            # 2D cells: 5 N bins × 3 frac bins
            sub["N_qbin"] = pd.qcut(sub["N"], 5, duplicates="drop")
            sub["frac_qbin"] = pd.qcut(sub["frac"], 3, duplicates="drop")

            # Local linear fit within each cell
            sub["thr_resid"] = np.nan
            for (nb, fb), idx in sub.groupby(["N_qbin", "frac_qbin"], observed=True).groups.items():
                cell = sub.loc[idx]
                if len(cell) < 5: continue
                X = np.column_stack([
                    np.ones(len(cell)),
                    cell["frac"].values,
                    cell["inv_sqrt_N"].values,
                ])
                y = cell["threshold"].values
                try:
                    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    sub.loc[idx, "thr_resid"] = y - X @ coeffs
                except np.linalg.LinAlgError:
                    sub.loc[idx, "thr_resid"] = y - y.mean()

            sub = sub.dropna(subset=["thr_resid"])

            # One σ per cell, color by frac
            frac_groups = sorted(sub["frac_qbin"].dropna().unique())
            frac_colors = N_COLORS_3[:len(frac_groups)]
            all_x, all_y = [], []

            for fq, fc in zip(frac_groups, frac_colors):
                frac_sub = sub[sub["frac_qbin"] == fq]
                n_groups = frac_sub.groupby("N_qbin", observed=True)

                x_c, y_c, y_e = [], [], []
                for name, cell in n_groups:
                    if len(cell) < 5: continue
                    sig = cell["thr_resid"].std()
                    sig_err = sig / np.sqrt(2 * (len(cell) - 1))
                    x_val = cell["inv_sqrt_Nf"].mean()
                    x_c.append(x_val)
                    y_c.append(sig)
                    y_e.append(sig_err)
                    all_x.append(x_val)
                    all_y.append(sig)

                x_c, y_c, y_e = np.array(x_c), np.array(y_c), np.array(y_e)
                if len(x_c) > 0:
                    ax.errorbar(x_c, y_c, yerr=y_e, fmt='o', ms=5, capsize=2,
                               color=fc, alpha=0.8,
                               label=f"frac∈{fq}" if row_idx == 0 else None)
                    global_xmax = max(global_xmax, x_c.max())
                    global_ymax = max(global_ymax, (y_c + y_e).max())

            # Fit: σ = a/√(N×frac) + b
            if len(all_x) > 3:
                all_x_arr, all_y_arr = np.array(all_x), np.array(all_y)
                slope, intercept, r_lin, _, _ = sp_stats.linregress(all_x_arr, all_y_arr)
                xfit = np.linspace(0, all_x_arr.max() * 1.1, 50)
                ax.plot(xfit, slope * xfit + intercept, 'k--', lw=1.5, alpha=0.8)

                fit_table_rows.append([method, f"{dx_mean:.3f}",
                                      f"{slope:.4f}", f"{intercept:.1e}", f"{r_lin**2:.3f}"])

            # Labels
            if col_idx == 0:
                ax.set_ylabel(f"{METHOD_LABELS.get(method, method)}\nσ(thr − fit)",
                             fontsize=9)
            if row_idx == 0:
                ax.set_title(f"⟨Δx⟩={dx_mean:.3f}", fontsize=10)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7)

    # Shared limits
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, global_xmax * 1.15)
            ax.set_ylim(0, global_ymax * 1.15)

    for col_idx in range(n_panels):
        axes[-1, col_idx].set_xlabel("1/√(N×frac)", fontsize=9)

    suffix = f"_{subset_key}" if subset_key != "all" else ""
    savefig(fig, os.path.join(output_dir, f"s2_threshold_stability{suffix}"),
            f"S2: Threshold Stability ({subset_label})")

    if fit_table_rows:
        report(f"\nS2 fit results ({subset_label}): σ(thr_resid) = slope/√(N×frac) + intercept, per Δx bin")
        report_table(["Method", "⟨Δx⟩", "slope", "intercept", "R²"], fit_table_rows)


# ===================================================================
# S3: PDF Estimator Bias
# ===================================================================

def figure_s3(obs, methods, output_dir, col_suffix="", label_suffix=""):
    """S3: RMS(pdf_emp/pdf_true - 1) vs 1/√λ where λ = N×pdf×Δx.

    Combined figure: rows = methods, cols = Δx bins.
    Shared x/y range so methods are directly comparable.
    All pdf_true colors should collapse if Poisson dominates.
    Model: RMS = √(a²/λ + b²). Fit a, b globally per method.

    col_suffix: "" for all bins, "_noedge" for edge-excluded.
    """
    n_dx_bins = 4

    # Get common Δx bins from all data
    d_all = obs.copy()
    d_all["dx_bin"] = pd.qcut(d_all["dx"], n_dx_bins, duplicates="drop")
    dx_groups = sorted(d_all["dx_bin"].dropna().unique())
    n_panels = len(dx_groups)
    n_methods = len(methods)

    if n_panels == 0:
        report("  S3: no dx groups — skipping")
        return

    fig, axes = plt.subplots(n_methods, n_panels, figsize=(5 * n_panels, 5 * n_methods),
                             sharex=True, sharey=True)
    if n_methods == 1: axes = axes.reshape(1, -1)
    if n_panels == 1: axes = axes.reshape(-1, 1)

    fig.suptitle(f"S3: PDF Estimator Bias{label_suffix} — RMS(pdf_emp/pdf_true − 1) vs 1/√λ\n"
                "λ = N×⟨pdf⟩×Δx — all colors should collapse if Poisson dominates\n"
                "Model: RMS = √(a²/λ + b²),  a=Poisson coeff (expect ~1), b=discretization floor",
                fontsize=11)

    # Compute global x/y range across all data
    global_xmax = 0
    global_ymax = 0

    fit_table_rows = []

    for row_idx, method in enumerate(methods):
        d = d_all[d_all["method"] == method].copy()
        if len(d) == 0: continue

        # Collect all points for global fit
        all_inv_sqrt_lam = []
        all_rms = []

        for col_idx, dx_grp in enumerate(dx_groups):
            ax = axes[row_idx, col_idx]
            sub_dx = d[d["dx_bin"] == dx_grp].copy()
            dx_mean = sub_dx["dx"].mean()

            for (plo, phi), pc, pl in zip(PDF_BINS, PDF_COLORS, PDF_LABELS):
                rms_col = f"rms_pdf{col_suffix}_{plo:.2f}_{phi:.2f}"
                lam_col = f"lambda{col_suffix}_{plo:.2f}_{phi:.2f}"
                if rms_col not in sub_dx.columns or lam_col not in sub_dx.columns:
                    continue

                sub = sub_dx.dropna(subset=[rms_col])
                if len(sub) == 0: continue

                inv_sqrt_lam = 1.0 / np.sqrt(sub[lam_col].values.astype(float))
                rms_vals = sub[rms_col].values.astype(float)

                sub = sub.copy()
                sub["isl"] = inv_sqrt_lam
                sub["rms_val"] = rms_vals
                try:
                    sub["isl_bin"] = pd.qcut(sub["isl"], 5, duplicates="drop")
                except ValueError:
                    continue

                grouped = sub.groupby("isl_bin", observed=True)
                x_c, y_m, y_e = [], [], []
                for name, grp in grouped:
                    if len(grp) < 3: continue
                    x_c.append(grp["isl"].mean())
                    y_m.append(grp["rms_val"].mean())
                    y_e.append(grp["rms_val"].std() / np.sqrt(len(grp)))

                x_c, y_m, y_e = np.array(x_c), np.array(y_m), np.array(y_e)
                if len(x_c) > 0:
                    ax.errorbar(x_c, y_m, yerr=y_e, fmt='o-', ms=5, capsize=2,
                               color=pc, label=pl, alpha=0.8)
                    global_xmax = max(global_xmax, x_c.max())
                    global_ymax = max(global_ymax, (y_m + y_e).max())

                valid = np.isfinite(inv_sqrt_lam) & np.isfinite(rms_vals)
                all_inv_sqrt_lam.extend(inv_sqrt_lam[valid])
                all_rms.extend(rms_vals[valid])

            # Row label on left column
            if col_idx == 0:
                ax.set_ylabel(f"{METHOD_LABELS.get(method, method)}\nRMS(pdf/pdf_true − 1)",
                             fontsize=9)

            # Column label on top row
            if row_idx == 0:
                ax.set_title(f"⟨Δx⟩={dx_mean:.3f}", fontsize=10)

            ax.legend(fontsize=8)

        # Global fit for this method
        ax_fit_x = np.array(all_inv_sqrt_lam)
        ax_fit_y = np.array(all_rms)
        valid = np.isfinite(ax_fit_x) & np.isfinite(ax_fit_y) & (ax_fit_x > 0)
        ax_fit_x, ax_fit_y = ax_fit_x[valid], ax_fit_y[valid]

        a_fit, b_fit, r2_1p, r2_2p = np.nan, np.nan, np.nan, np.nan
        if len(ax_fit_x) > 10:
            # 1-param: y = a*x
            a_fit = np.sum(ax_fit_y * ax_fit_x) / np.sum(ax_fit_x ** 2)
            ss_res = np.sum((ax_fit_y - a_fit * ax_fit_x) ** 2)
            ss_tot = np.sum((ax_fit_y - ax_fit_y.mean()) ** 2)
            r2_1p = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            # 2-param: y = √(a²x² + b²)
            try:
                def model_2p(x, a, b):
                    return np.sqrt((a * x) ** 2 + b ** 2)
                popt, _ = curve_fit(model_2p, ax_fit_x, ax_fit_y, p0=[a_fit, 0.001],
                                   maxfev=5000)
                a_2p, b_2p = popt
                y_pred = model_2p(ax_fit_x, a_2p, b_2p)
                ss_res2 = np.sum((ax_fit_y - y_pred) ** 2)
                r2_2p = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0
                b_fit = b_2p
            except Exception:
                b_fit = np.nan
                r2_2p = np.nan

            # Overlay fit on all panels for this method
            for col_idx in range(n_panels):
                ax = axes[row_idx, col_idx]
                xfit = np.linspace(0, global_xmax * 1.1, 50)
                ax.plot(xfit, a_fit * xfit, 'k--', lw=1.5, alpha=0.7)
                if np.isfinite(b_fit):
                    ax.plot(xfit, np.sqrt((a_fit * xfit) ** 2 + b_fit ** 2),
                           'r:', lw=1.5, alpha=0.6)

            fit_table_rows.append([method, f"{a_fit:.4f}", f"{r2_1p:.4f}",
                                  f"{b_fit:.6f}" if np.isfinite(b_fit) else "—",
                                  f"{r2_2p:.4f}" if np.isfinite(r2_2p) else "—"])

            report(f"\n  S3{col_suffix} {method} global fit:")
            report(f"    Poisson: a={a_fit:.4f}, R²={r2_1p:.4f}")
            if np.isfinite(b_fit):
                report(f"    2-param: a={a_fit:.4f}, b={b_fit:.6f}, R²={r2_2p:.4f}")

    # Set shared limits
    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, global_xmax * 1.1)
            ax.set_ylim(0, global_ymax * 1.1)

    # Bottom row x-labels
    for col_idx in range(n_panels):
        axes[-1, col_idx].set_xlabel("1/√λ  (λ = N×pdf×Δx)", fontsize=9)

    # Add fit legend to last panel of each row
    for row_idx, method in enumerate(methods):
        ax = axes[row_idx, -1]
        if fit_table_rows and row_idx < len(fit_table_rows):
            fr = fit_table_rows[row_idx]
            ax.plot([], [], 'k--', lw=1.5, label=f"a/√λ: a={fr[1]}, R²={fr[2]}")
            if fr[3] != "—":
                ax.plot([], [], 'r:', lw=1.5, label=f"√(a²/λ+b²): b={fr[3]}")
            ax.legend(fontsize=8)

    file_suffix = col_suffix if col_suffix else ""
    savefig(fig, os.path.join(output_dir, f"s3_pdf_bias{file_suffix}"),
            f"S3: PDF Bias{label_suffix}")

    if fit_table_rows:
        report(f"\nS3 fit summary{label_suffix}: RMS = √(a²/λ + b²)")
        report_table(["Method", "a", "R²(Poisson)", "b", "R²(2-param)"], fit_table_rows)


# ===================================================================
# S3b: PDF Estimator Bias vs position
# ===================================================================

def figure_s3b(obs, methods, output_dir, scan=None, params=None, meta=None):
    """S3b: PDF estimator bias as a function of x position.

    3 rows × 2 cols: col 0 = all bins, col 1 = excluding edge bins.
    Row 0: bias (mean) vs x
    Row 1: RMS vs x
    Row 2: RMS vs per-point dx
    Color = method. If isEdge column missing, col 1 shows same as col 0.
    """
    if scan is None or params is None:
        report("  S3b: no scan tree — skipping")
        return

    has_edge = "isEdge" in scan.columns

    x_lo = meta["x_lo"] if meta else -3 * SIGMA
    x_hi = meta["x_hi"] if meta else 3 * SIGMA
    x_bins = np.linspace(x_lo, x_hi, 31)
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    n_xbins = len(x_centers)

    # Collect per-point data
    # key: (method, subset) -> profiles
    profiles = {}  # (method, subset) -> (bias_mean, bias_rms, count)
    dx_data = {}   # (method, subset) -> (dx_arr, bias_arr)

    for method in methods:
        pdf_col = f"{method}_pdf"
        is_col = f"{method}_is_sampled"

        all_x, all_bias, all_dx, all_edge = [], [], [], []

        for _, prow in params.iterrows():
            iteration = int(prow["iteration"])
            mask = (scan["iteration"] == iteration) & (scan[is_col] == 1)
            data = scan[mask]
            if len(data) == 0: continue

            pdf = data[pdf_col].values.astype(np.float64)
            pdf_true = data["pdf_true"].values.astype(np.float64)
            x = data["x"].values.astype(np.float64)
            dx = data["dx"].values.astype(np.float64)
            edge = data["isEdge"].values.astype(np.int8) if has_edge else np.zeros(len(data), dtype=np.int8)

            finite = np.isfinite(pdf) & (pdf > 0) & (pdf_true > 0.001)
            if finite.sum() < 10: continue

            bias = pdf[finite] / pdf_true[finite] - 1.0
            all_x.extend(x[finite])
            all_bias.extend(bias)
            all_dx.extend(dx[finite])
            all_edge.extend(edge[finite])

        all_x = np.array(all_x)
        all_bias = np.array(all_bias)
        all_dx = np.array(all_dx)
        all_edge = np.array(all_edge)

        for subset_name, subset_mask in [("all", np.ones(len(all_x), dtype=bool)),
                                          ("no_edge", all_edge == 0)]:
            sx, sb, sd = all_x[subset_mask], all_bias[subset_mask], all_dx[subset_mask]
            if len(sx) < 100:
                continue

            bias_mean = np.full(n_xbins, np.nan)
            bias_rms = np.full(n_xbins, np.nan)
            bias_count = np.full(n_xbins, 0)
            bidx = np.clip(np.digitize(sx, x_bins) - 1, 0, n_xbins - 1)

            for i in range(n_xbins):
                sel = bidx == i
                if sel.sum() > 50:
                    bias_mean[i] = np.mean(sb[sel])
                    bias_rms[i] = np.sqrt(np.mean(sb[sel] ** 2))
                    bias_count[i] = sel.sum()

            profiles[(method, subset_name)] = (bias_mean, bias_rms, bias_count)
            dx_data[(method, subset_name)] = (sd, sb)

    # --- Plot: 2 rows × 4 cols ---
    # Row 0 = all bins, Row 1 = excluding edge bins
    # Cols: bias vs x, RMS vs x, bias vs dx, RMS vs dx
    subsets = [("all", "All bins"), ("no_edge", "Excluding edge bins")]
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))

    fig.suptitle("S3b: PDF Estimator Bias — Row 0: all bins, Row 1: excluding edge bins\n"
                "Cols: ⟨bias⟩ vs x,  RMS vs x,  ⟨bias⟩ vs dx,  RMS vs dx",
                fontsize=11)

    method_colors = {"smooth": "#d62728", "smooth_v5": "#1f77b4"}

    col_labels = ["⟨bias⟩ vs x", "RMS vs x", "⟨bias⟩ vs dx", "RMS vs dx"]

    for row_idx, (subset_key, subset_label) in enumerate(subsets):
        # Col 0: bias vs x
        ax = axes[row_idx, 0]
        for method in methods:
            key = (method, subset_key)
            if key not in profiles: continue
            bias_mean, _, count = profiles[key]
            ok = count > 50
            mc = method_colors.get(method, "gray")
            ax.plot(x_centers[ok], bias_mean[ok], 'o', ms=4, color=mc,
                   label=METHOD_LABELS.get(method, method), alpha=0.8)
        ax.axhline(0, color='red', ls='--', lw=1, alpha=0.5)
        ax.set_ylabel(subset_label, fontsize=10)
        if row_idx == 0:
            ax.set_title(col_labels[0], fontsize=9)
            ax.legend(fontsize=7)
        if row_idx == 1:
            ax.set_xlabel("x", fontsize=9)

        # Col 1: RMS vs x
        ax = axes[row_idx, 1]
        for method in methods:
            key = (method, subset_key)
            if key not in profiles: continue
            _, bias_rms, count = profiles[key]
            ok = count > 50
            mc = method_colors.get(method, "gray")
            ax.plot(x_centers[ok], bias_rms[ok], 'o', ms=4, color=mc,
                   label=METHOD_LABELS.get(method, method), alpha=0.8)
        ax.set_ylim(bottom=0)
        if row_idx == 0:
            ax.set_title(col_labels[1], fontsize=9)
        if row_idx == 1:
            ax.set_xlabel("x", fontsize=9)

        # Col 2: bias vs dx
        ax = axes[row_idx, 2]
        for method in methods:
            key = (method, subset_key)
            if key not in dx_data: continue
            sd, sb = dx_data[key]
            if len(sd) < 100: continue

            dx_edges_q = np.quantile(sd, np.linspace(0, 1, 21))
            dx_edges_q = np.unique(dx_edges_q)
            if len(dx_edges_q) < 3: continue
            dx_cntrs = 0.5 * (dx_edges_q[:-1] + dx_edges_q[1:])
            didx = np.clip(np.digitize(sd, dx_edges_q) - 1, 0, len(dx_cntrs) - 1)

            bias_list, dx_list = [], []
            for i in range(len(dx_cntrs)):
                sel = didx == i
                if sel.sum() > 50:
                    bias_list.append(np.mean(sb[sel]))
                    dx_list.append(dx_cntrs[i])

            mc = method_colors.get(method, "gray")
            ax.plot(dx_list, bias_list, 'o-', ms=4, color=mc,
                   label=METHOD_LABELS.get(method, method), alpha=0.8)
        ax.axhline(0, color='red', ls='--', lw=1, alpha=0.5)
        if row_idx == 0:
            ax.set_title(col_labels[2], fontsize=9)
        if row_idx == 1:
            ax.set_xlabel("dx", fontsize=9)

        # Col 3: RMS vs dx
        ax = axes[row_idx, 3]
        for method in methods:
            key = (method, subset_key)
            if key not in dx_data: continue
            sd, sb = dx_data[key]
            if len(sd) < 100: continue

            dx_edges_q = np.quantile(sd, np.linspace(0, 1, 21))
            dx_edges_q = np.unique(dx_edges_q)
            if len(dx_edges_q) < 3: continue
            dx_cntrs = 0.5 * (dx_edges_q[:-1] + dx_edges_q[1:])
            didx = np.clip(np.digitize(sd, dx_edges_q) - 1, 0, len(dx_cntrs) - 1)

            rms_list, dx_list = [], []
            for i in range(len(dx_cntrs)):
                sel = didx == i
                if sel.sum() > 50:
                    rms_list.append(np.sqrt(np.mean(sb[sel] ** 2)))
                    dx_list.append(dx_cntrs[i])

            mc = method_colors.get(method, "gray")
            ax.plot(dx_list, rms_list, 'o-', ms=4, color=mc,
                   label=METHOD_LABELS.get(method, method), alpha=0.8)
        ax.set_ylim(bottom=0)
        if row_idx == 0:
            ax.set_title(col_labels[3], fontsize=9)
        if row_idx == 1:
            ax.set_xlabel("dx", fontsize=9)

    savefig(fig, os.path.join(output_dir, "s3b_pdf_bias_vs_x"),
            "S3b: PDF Bias vs Position")

    # Report table: RMS per |x| category, both subsets
    s3b_rows = []
    for method in methods:
        for subset_key, subset_label in subsets:
            key = (method, subset_key)
            if key not in profiles: continue
            _, bias_rms, count = profiles[key]
            b1 = meta["b1"] if meta else 1.0
            b2 = meta["b2"] if meta else 2.0
            x_mid_val = meta["x_mid"] if meta else 0.0
            for (xlo, xhi, label) in [(0, b1, "core"), (b1, b2, "shoulder"), (b2, 999, "tail")]:
                sel = (np.abs(x_centers - x_mid_val) >= xlo) & (np.abs(x_centers - x_mid_val) < xhi) & (count > 50)
                if sel.sum() > 0:
                    rms_mean = np.mean(bias_rms[sel])
                    n_pts = int(np.sum(count[sel]))
                    s3b_rows.append([method, subset_label, label, f"{rms_mean:.4f}", str(n_pts)])

    if s3b_rows:
        report("\nS3b: PDF bias RMS by |x| region")
        report_table(["Method", "Subset", "|x| region", "⟨RMS⟩", "N_points"], s3b_rows)


# ===================================================================
# S4: Spectra Recovery
# ===================================================================

def _compute_per_xbin_ratios(scan, params, methods, meta=None):
    """Precompute per-iteration per-x-bin ratios. Heavy loop, run once."""
    x_lo = meta["x_lo"] if meta else -3 * SIGMA
    x_hi = meta["x_hi"] if meta else 3 * SIGMA
    hist_bins = np.linspace(x_lo, x_hi, 31)
    bin_centers = 0.5 * (hist_bins[:-1] + hist_bins[1:])
    n_bins = len(bin_centers)

    results = {}  # key: (method, iteration) -> ratio array

    for method in methods:
        pdf_col = f"{method}_pdf"
        thr_col = f"{method}_threshold"
        is_col = f"{method}_is_sampled"

        for _, prow in params.iterrows():
            iteration = int(prow["iteration"])
            N = int(prow["N"])

            mask = (scan["iteration"] == iteration) & (scan[is_col] == 1)
            data = scan[mask]
            if len(data) == 0: continue

            x = data["x"].values.astype(np.float64)
            pdf = data[pdf_col].values.astype(np.float64)
            thr = data[thr_col].values.astype(np.float64)
            threshold = thr[0]

            w_ht = np.maximum(1.0, pdf / threshold)

            exp_counts = expected_counts_from_profile(N, hist_bins, meta) if meta else np.ones(n_bins)

            reco, _ = np.histogram(x, bins=hist_bins, weights=w_ht)

            ratio = np.full(n_bins, np.nan)
            good = exp_counts > 10
            ratio[good] = reco[good] / exp_counts[good]

            results[(method, iteration)] = ratio

    return hist_bins, bin_centers, results


def _s4_plot_panel(axes_bias, axes_sig, bin_centers, bin_width,
                   ratios_dict, sub_params, method, obs,
                   color, label, meta=None):
    """Plot one group's bias and sigma (no connecting lines). Return model data."""
    all_ratios = []
    for _, prow in sub_params.iterrows():
        iteration = int(prow["iteration"])
        key = (method, iteration)
        if key in ratios_dict:
            all_ratios.append(ratios_dict[key])

    if len(all_ratios) < 5:
        return None

    ratios_arr = np.array(all_ratios)
    with np.errstate(all='ignore'):
        bias = np.nanmean(ratios_arr - 1.0, axis=0)
        sigma = np.nanstd(ratios_arr, axis=0)
        n_valid = np.sum(~np.isnan(ratios_arr), axis=0)

    ok = n_valid >= 5

    # Bias (row 0) — markers only, no lines
    axes_bias.errorbar(bin_centers[ok], bias[ok],
                      yerr=sigma[ok] / np.sqrt(n_valid[ok]),
                      fmt='o', ms=3, capsize=1, color=color,
                      label=label, alpha=0.8)

    # Sigma with error bars (row 1) — markers only
    sigma_err = sigma / np.sqrt(2 * np.maximum(n_valid - 1, 1))
    axes_sig.errorbar(bin_centers[ok], sigma[ok], yerr=sigma_err[ok],
                     fmt='o', ms=3, capsize=1, color=color,
                     label=label, alpha=0.8)

    # Model: σ_model = √(⟨σ_model_i²⟩) where σ_model_i = 1/√(N_i × min(f, thr_i) × Δx)
    iter_list = sub_params["iteration"].values
    obs_sub = obs[(obs["method"] == method) & obs["iteration"].isin(iter_list)]
    N_vals = sub_params["N"].values.astype(float)
    thr_vals = obs_sub["threshold"].values if len(obs_sub) > 0 else np.array([])

    sigma_model = np.full_like(sigma, np.nan)
    mean_ratio = np.nan
    mean_ratio_err = np.nan
    if len(thr_vals) > 0 and len(N_vals) > 0:
        f_true_vals = pdf_true_at(bin_centers, meta) if meta else gaussian_pdf(bin_centers)
        n_iter = min(len(N_vals), len(thr_vals))
        sig_models_i = np.zeros((n_iter, len(bin_centers)))
        for i in range(n_iter):
            sig_models_i[i] = np.where(
                f_true_vals > thr_vals[i],
                1.0 / np.sqrt(N_vals[i] * thr_vals[i] * bin_width + 1e-30),
                1.0 / np.sqrt(N_vals[i] * f_true_vals * bin_width + 1e-30)
            )
        sigma_model = np.sqrt(np.mean(sig_models_i ** 2, axis=0))
        axes_sig.plot(bin_centers[ok], sigma_model[ok], '--', color=color,
                     alpha=0.4, lw=1.5)

        # Mean ratio σ_meas / σ_model (core only)
        core_half = meta["b1"] if meta else 2 * SIGMA
        x_mid = meta["x_mid"] if meta else 0.0
        core = ok & (np.abs(bin_centers - x_mid) < core_half) & np.isfinite(sigma_model) & (sigma_model > 0)
        if core.sum() > 3:
            ratios_core = sigma[core] / sigma_model[core]
            mean_ratio = np.mean(ratios_core)
            mean_ratio_err = np.std(ratios_core) / np.sqrt(len(ratios_core))

    return {"sigma": sigma, "sigma_model": sigma_model, "ok": ok,
            "n_valid": n_valid, "mean_ratio": mean_ratio, "mean_ratio_err": mean_ratio_err}


def _s4_compute_sigma(ratios_dict, sub_params, method, obs,
                      bin_centers, bin_width, meta=None):
    """Compute σ and σ_model for a group without plotting. For S4c scatter."""
    all_ratios = []
    for _, prow in sub_params.iterrows():
        key = (method, int(prow["iteration"]))
        if key in ratios_dict:
            all_ratios.append(ratios_dict[key])
    if len(all_ratios) < 5:
        return None

    ratios_arr = np.array(all_ratios)
    with np.errstate(all='ignore'):
        sigma = np.nanstd(ratios_arr, axis=0)
        n_valid = np.sum(~np.isnan(ratios_arr), axis=0)
    ok = n_valid >= 5

    iter_list = sub_params["iteration"].values
    obs_sub = obs[(obs["method"] == method) & obs["iteration"].isin(iter_list)]
    N_vals = sub_params["N"].values.astype(float)
    thr_vals = obs_sub["threshold"].values if len(obs_sub) > 0 else np.array([])

    sigma_model = np.full_like(sigma, np.nan)
    if len(thr_vals) > 0 and len(N_vals) > 0:
        f_true_vals = pdf_true_at(bin_centers, meta) if meta else gaussian_pdf(bin_centers)

        # Per-iteration σ_model_i, then σ_model = √(⟨σ_model_i²⟩)
        n_iter = min(len(N_vals), len(thr_vals))
        sig_models_i = np.zeros((n_iter, len(bin_centers)))
        for i in range(n_iter):
            sig_models_i[i] = np.where(
                f_true_vals > thr_vals[i],
                1.0 / np.sqrt(N_vals[i] * thr_vals[i] * bin_width + 1e-30),
                1.0 / np.sqrt(N_vals[i] * f_true_vals * bin_width + 1e-30)
            )
        sigma_model = np.sqrt(np.mean(sig_models_i ** 2, axis=0))

    return {"sigma": sigma, "sigma_model": sigma_model, "ok": ok, "n_valid": n_valid}


def figure_s4(obs, methods, output_dir, scan=None, params=None, meta=None):
    """S4: Two variants of spectra recovery per x-bin.
    S4a: cols = frac bins, color = N bins
    S4b: cols = Δx bins, color = N_sampled bins
    """
    if scan is None or params is None:
        _figure_s4_simple(obs, methods, output_dir)
        return

    report("  Computing per-x-bin ratios...")
    t_s4 = time.time()
    hist_bins, bin_centers, ratios_dict = _compute_per_xbin_ratios(scan, params, methods, meta=meta)
    bin_width = hist_bins[1] - hist_bins[0]
    report(f"  Per-x-bin ratios: {time.time() - t_s4:.1f}s")

    params = params.copy()

    # --- S4a: cols = frac bins, color = N bins ---
    n_frac_bins = 3
    params["frac_qbin"] = pd.qcut(params["frac"], n_frac_bins, duplicates="drop")
    frac_groups = sorted(params["frac_qbin"].dropna().unique())

    params["logN"] = np.log10(params["N"].astype(float))
    logN_edges = np.quantile(params["logN"], [0, 1/3, 2/3, 1.0])
    logN_edges[0] -= 0.01; logN_edges[-1] += 0.01
    n_labels = [f"N:[{10**logN_edges[i]:.0f},{10**logN_edges[i+1]:.0f}]"
                for i in range(3)]
    params["N_bin"] = pd.cut(params["logN"], bins=logN_edges, labels=n_labels,
                            include_lowest=True)

    for method in methods:
        fig, axes = plt.subplots(2, len(frac_groups), figsize=(5 * len(frac_groups), 8),
                                 sharex=True)
        if len(frac_groups) == 1: axes = axes.reshape(-1, 1)

        fig.suptitle(f"S4a: Spectra Recovery — {METHOD_LABELS.get(method, method)}\n"
                    "Row 0: bias, Row 1: σ (with error bars + model dashed)\n"
                    "Cols = frac bins, Color = N bins", fontsize=10)

        for col_idx, fq in enumerate(frac_groups):
            frac_params = params[params["frac_qbin"] == fq]
            frac_mean = frac_params["frac"].mean()

            for n_label, n_color in zip(n_labels, N_COLORS_3):
                sub = frac_params[frac_params["N_bin"] == n_label]
                if len(sub) == 0: continue
                _s4_plot_panel(axes[0, col_idx], axes[1, col_idx],
                              bin_centers, bin_width, ratios_dict,
                              sub, method, obs, n_color, n_label, meta=meta)

            axes[0, col_idx].set_title(f"⟨frac⟩={frac_mean:.3f}", fontsize=10)
            axes[0, col_idx].axhline(0, color='red', ls='--', lw=1, alpha=0.5)
            axes[1, col_idx].set_xlabel("x", fontsize=9)
            if col_idx == 0:
                axes[0, col_idx].set_ylabel("⟨ratio−1⟩ (bias)", fontsize=9)
                axes[1, col_idx].set_ylabel("σ(ratio)", fontsize=9)
                axes[0, col_idx].legend(fontsize=7)

        savefig(fig, os.path.join(output_dir, f"s4a_spectra_frac_{method}"),
                f"S4a: Spectra by frac ({method})")

    # --- S4b: cols = Δx bins, color = N_sampled bins (2 rows) ---
    n_dx_bins = 3
    params["dx_bin"] = pd.qcut(params["dx"], n_dx_bins, duplicates="drop")
    dx_groups = sorted(params["dx_bin"].dropna().unique())

    if len(dx_groups) == 0:
        report("  WARNING: no dx groups — skipping S4b/S4c")
        return

    # Collect all model data for S4c
    s4c_data = {}
    s4b_ratio_table = []  # for summary report

    for method in methods:
        obs_m = obs[obs["method"] == method][["iteration", "N_sampled"]].copy()
        params_m = params.merge(obs_m, on="iteration", how="left")

        ns_edges = np.quantile(params_m["N_sampled"].dropna(), [0, 1/3, 2/3, 1.0])
        ns_edges[0] -= 1; ns_edges[-1] += 1
        ns_labels = [f"Ns:[{ns_edges[i]:.0f},{ns_edges[i+1]:.0f}]" for i in range(3)]
        params_m["Ns_bin"] = pd.cut(params_m["N_sampled"], bins=ns_edges,
                                    labels=ns_labels, include_lowest=True)

        fig, axes = plt.subplots(2, len(dx_groups), figsize=(5 * len(dx_groups), 8),
                                 sharex=True)
        if len(dx_groups) == 1: axes = axes.reshape(-1, 1)

        fig.suptitle(f"S4b: Spectra Recovery — {METHOD_LABELS.get(method, method)}\n"
                    "Row 0: bias, Row 1: σ (markers + model dashed)\n"
                    "Model: σ = √(⟨1/(N×thr)⟩/Δx) dense, √(⟨1/N⟩/(f×Δx)) sparse",
                    fontsize=10)

        for col_idx, dx_grp in enumerate(dx_groups):
            dx_sub = params_m[params_m["dx_bin"] == dx_grp]
            dx_mean = dx_sub["dx"].mean()

            for ns_label, ns_color in zip(ns_labels, N_COLORS_3):
                sub = dx_sub[dx_sub["Ns_bin"] == ns_label]
                if len(sub) == 0: continue
                result = _s4_plot_panel(axes[0, col_idx], axes[1, col_idx],
                              bin_centers, bin_width, ratios_dict,
                              sub, method, obs, ns_color, ns_label, meta=meta)
                if result is not None:
                    s4c_data[(method, col_idx, ns_label)] = result
                    mr = result["mean_ratio"]
                    mre = result["mean_ratio_err"]
                    if np.isfinite(mr):
                        s4b_ratio_table.append([method, f"{dx_mean:.3f}", ns_label,
                                               f"{mr:.3f}±{mre:.3f}"])

            axes[0, col_idx].set_title(f"⟨Δx⟩={dx_mean:.3f}", fontsize=10)
            axes[0, col_idx].axhline(0, color='red', ls='--', lw=1, alpha=0.5)
            axes[1, col_idx].set_xlabel("x", fontsize=9)
            if col_idx == 0:
                axes[0, col_idx].set_ylabel("⟨ratio−1⟩ (bias)", fontsize=9)
                axes[1, col_idx].set_ylabel("σ(ratio)", fontsize=9)
                axes[0, col_idx].legend(fontsize=7)

        savefig(fig, os.path.join(output_dir, f"s4b_spectra_dx_{method}"),
                f"S4b: Spectra by Δx ({method})")

    if s4b_ratio_table:
        report("\nS4b model agreement: ⟨σ_meas/σ_model⟩ in core (|x|<2σ)")
        report_table(["Method", "⟨Δx⟩", "N_sampled bin", "⟨σ/σ_model⟩"], s4b_ratio_table)

    # --- S4c: 3 cols (Dx) x 3 rows: ratio vs x, scatter, pull histogram ---
    b1 = meta["b1"] if meta else 1.0
    b2 = meta["b2"] if meta else 2.0
    ABS_X_BINS = [(0, b1, "core"), (b1, b2, "shoulder"), (b2, 999, "tail")]
    ABS_X_COLORS = ["#1b9e77", "#d95f02", "#7570b3"]

    s4c_fit_table = []
    s4c_pull_table = []

    for method in methods:
        obs_m = obs[obs["method"] == method][["iteration", "N_sampled", "threshold"]].copy()
        params_m = params.merge(obs_m, on="iteration", how="left")

        n_ns_fine = 6
        ns_fine_edges = np.quantile(params_m["N_sampled"].dropna(), np.linspace(0, 1, n_ns_fine + 1))
        ns_fine_edges[0] -= 1; ns_fine_edges[-1] += 1
        ns_fine_labels = [f"Ns:{i}" for i in range(n_ns_fine)]
        params_m["Ns_fine"] = pd.cut(params_m["N_sampled"], bins=ns_fine_edges,
                                     labels=ns_fine_labels, include_lowest=True)

        ns_edges = np.quantile(params_m["N_sampled"].dropna(), [0, 1/3, 2/3, 1.0])
        ns_edges[0] -= 1; ns_edges[-1] += 1
        ns_labels = [f"Ns:[{ns_edges[i]:.0f},{ns_edges[i+1]:.0f}]" for i in range(3)]

        fig, axes_c = plt.subplots(3, len(dx_groups), figsize=(5 * len(dx_groups), 12))
        if len(dx_groups) == 1: axes_c = axes_c.reshape(-1, 1)

        fig.suptitle(f"S4c: sigma Scaling - {METHOD_LABELS.get(method, method)}\n"
                    "Row 0: sigma_meas/sigma_model vs x.  Row 1: scatter.  Row 2: pull.\n"
                    "Scatter color = |x| category (core/shoulder/tail)", fontsize=10)

        global_sig_meas, global_sig_model = [], []

        for col_idx, dx_grp in enumerate(dx_groups):
            dx_mean = params[params["dx_bin"] == dx_grp]["dx"].mean()
            ax_ratio = axes_c[0, col_idx]
            ax_scatter = axes_c[1, col_idx]
            ax_pull = axes_c[2, col_idx]

            # Row 0: ratio per Ns group
            for ns_label, ns_color in zip(ns_labels, N_COLORS_3):
                key = (method, col_idx, ns_label)
                if key not in s4c_data: continue
                d = s4c_data[key]
                sigma, sigma_model, ok, n_valid = d["sigma"], d["sigma_model"], d["ok"], d["n_valid"]

                valid = ok & np.isfinite(sigma_model) & (sigma_model > 0)
                if valid.sum() == 0: continue

                ratio = sigma[valid] / sigma_model[valid]
                ratio_err = sigma[valid] / (sigma_model[valid] * np.sqrt(2 * np.maximum(n_valid[valid] - 1, 1)))

                ax_ratio.errorbar(bin_centers[valid], ratio, yerr=ratio_err,
                                 fmt='o', ms=3, capsize=1, color=ns_color,
                                 label=ns_label, alpha=0.8)

                mr = d["mean_ratio"]
                if np.isfinite(mr):
                    ax_ratio.axhline(mr, color=ns_color, ls=':', lw=1, alpha=0.5)

            ax_ratio.axhline(1.0, color='red', ls='--', lw=1, alpha=0.5)
            ax_ratio.set_title(f"<Dx>={dx_mean:.3f}", fontsize=10)
            if col_idx == 0:
                ax_ratio.set_ylabel("sigma_meas / sigma_model", fontsize=9)
                ax_ratio.legend(fontsize=7)

            # Row 1 + Row 2: scatter and pull from 6 fine Ns bins
            dx_sub = params_m[params_m["dx_bin"] == dx_grp]

            all_sig_meas, all_sig_model, all_abs_x, all_n_valid = [], [], [], []

            for ns_fine_label in ns_fine_labels:
                sub = dx_sub[dx_sub["Ns_fine"] == ns_fine_label]
                if len(sub) < 5: continue

                result = _s4_compute_sigma(ratios_dict, sub, method, obs,
                                           bin_centers, bin_width, meta=meta)
                if result is None: continue

                sig, sig_mod, ok_r, nv = result["sigma"], result["sigma_model"], result["ok"], result["n_valid"]
                valid = ok_r & np.isfinite(sig_mod) & (sig_mod > 0)
                all_sig_meas.extend(sig[valid])
                all_sig_model.extend(sig_mod[valid])
                x_mid_s4c = meta["x_mid"] if meta else 0.0
                all_abs_x.extend(np.abs(bin_centers[valid] - x_mid_s4c))
                all_n_valid.extend(nv[valid])

            all_sig_meas = np.array(all_sig_meas)
            all_sig_model = np.array(all_sig_model)
            all_abs_x = np.array(all_abs_x)
            all_n_valid = np.array(all_n_valid, dtype=float)

            if len(all_sig_meas) > 5:
                # Scatter with 3 discrete |x| categories
                for (xlo, xhi, xlabel), xcolor in zip(ABS_X_BINS, ABS_X_COLORS):
                    mask = (all_abs_x >= xlo) & (all_abs_x < xhi)
                    if mask.sum() == 0: continue
                    ax_scatter.scatter(all_sig_model[mask], all_sig_meas[mask],
                                      s=15, alpha=0.6, color=xcolor, label=xlabel)

                maxval = max(all_sig_meas.max(), all_sig_model.max()) * 1.1
                ax_scatter.plot([0, maxval], [0, maxval], 'k--', lw=1.5)

                # Fit: y = a*x
                a_fit = np.sum(all_sig_meas * all_sig_model) / np.sum(all_sig_model ** 2)
                y_pred = a_fit * all_sig_model
                ss_res = np.sum((all_sig_meas - y_pred) ** 2)
                ss_tot = np.sum((all_sig_meas - all_sig_meas.mean()) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                ax_scatter.plot([0, maxval], [0, a_fit * maxval], 'r-', lw=1,
                               alpha=0.7, label=f"slope={a_fit:.3f}, R2={r2:.3f}")
                ax_scatter.legend(fontsize=6)
                ax_scatter.set_xlim(0, maxval); ax_scatter.set_ylim(0, maxval)
                ax_scatter.set_aspect('equal')

                s4c_fit_table.append([method, f"{dx_mean:.3f}", f"{a_fit:.4f}", f"{r2:.4f}"])
                global_sig_meas.extend(all_sig_meas)
                global_sig_model.extend(all_sig_model)

                # Pull: (sigma_meas - sigma_model) / sigma_err
                # sigma_err = sigma_meas / sqrt(2*(n-1)) — standard chi^2 error on std
                # sigma_model already uses √(⟨σ_model_i²⟩) via _s4_compute_sigma
                sigma_err = all_sig_meas / np.sqrt(2 * np.maximum(all_n_valid - 1, 1))
                pull = (all_sig_meas - all_sig_model) / np.maximum(sigma_err, 1e-30)

                # Pull histogram per |x| category
                for (xlo, xhi, xlabel), xcolor in zip(ABS_X_BINS, ABS_X_COLORS):
                    mask = (all_abs_x >= xlo) & (all_abs_x < xhi)
                    if mask.sum() < 3: continue
                    ax_pull.hist(pull[mask], bins=20, range=(-5, 5),
                                alpha=0.4, color=xcolor, label=xlabel)

                # Annotate with mean and RMS per category
                pull_stats_text = []
                for (xlo, xhi, xlabel), xcolor in zip(ABS_X_BINS, ABS_X_COLORS):
                    mask = (all_abs_x >= xlo) & (all_abs_x < xhi)
                    if mask.sum() < 3: continue
                    pm = np.mean(pull[mask])
                    prms = np.std(pull[mask])
                    pull_stats_text.append(f"{xlabel}: μ={pm:.2f}, RMS={prms:.2f}")
                    s4c_pull_table.append([method, f"{dx_mean:.3f}", xlabel,
                                          f"{pm:.2f}", f"{prms:.2f}", str(mask.sum())])

                ax_pull.text(0.02, 0.98, "\n".join(pull_stats_text),
                           transform=ax_pull.transAxes, fontsize=7,
                           va='top', ha='left', family='monospace')

            ax_scatter.set_xlabel("sigma_model", fontsize=9)
            ax_pull.set_xlabel("pull = (σ_meas−σ_model) / (σ_meas/√(2(n−1)))", fontsize=7)
            if col_idx == 0:
                ax_scatter.set_ylabel("sigma_measured", fontsize=9)
                ax_pull.set_ylabel("count", fontsize=9)

        savefig(fig, os.path.join(output_dir, f"s4c_sigma_scaling_{method}"),
                f"S4c: sigma Scaling ({method})")

        # Global fit
        gm, gmod = np.array(global_sig_meas), np.array(global_sig_model)
        if len(gm) > 10:
            a_g = np.sum(gm * gmod) / np.sum(gmod ** 2)
            ss_res = np.sum((gm - a_g * gmod) ** 2)
            ss_tot = np.sum((gm - gm.mean()) ** 2)
            r2_g = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            report(f"  S4c {method} global fit: slope={a_g:.4f}, R2={r2_g:.4f}")

    if s4c_fit_table:
        report("\nS4c fit results: sigma_meas = slope * sigma_model")
        report_table(["Method", "<Dx>", "slope", "R2"], s4c_fit_table)

    if s4c_pull_table:
        report("\nS4c pull statistics: pull = (σ_meas − σ_model) / (σ_meas/√(2(n−1)))")
        report_table(["Method", "<Dx>", "|x| bin", "mean", "RMS", "N_pts"], s4c_pull_table)

    # Summary
    fit_rows = []
    for method in methods:
        d = obs[obs["method"] == method]
        m = d["spectra_ratio_mean"].mean()
        s = d["spectra_ratio_mean"].std()
        fit_rows.append([method, f"{m:.4f}", f"{s:.4f}"])
    if fit_rows:
        report("\nS4 spectra recovery summary (per-iteration mean ratio):")
        report_table(["Method", "⟨ratio⟩", "σ"], fit_rows)


def _figure_s4_simple(obs, methods, output_dir):
    """Fallback S4 when scan tree not available."""
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6))
    if len(methods) == 1: axes = [axes]
    fig.suptitle("S4: Spectra Recovery (summary only — no scan tree)", fontsize=11)

    for ax, method in zip(axes, methods):
        d = obs[obs["method"] == method].copy()
        d = d.dropna(subset=["spectra_ratio_mean"])
        if len(d) == 0: continue

        n_labels = add_logN_bins(d, 3)
        add_frac_bins(d, 5)

        for n_label, n_color in zip(n_labels, N_COLORS_3):
            sub = d[d["N_bin"] == n_label]
            x_c, y_m, y_e = grouped_profile(sub, "frac", "spectra_ratio_mean", "frac_bin")
            if len(x_c) > 0:
                ax.errorbar(x_c, y_m, yerr=y_e, fmt='o-', ms=5, capsize=2,
                           color=n_color, label=n_label, alpha=0.8)

        ax.axhline(1.0, color='red', ls='--', lw=1.5)
        m = d["spectra_ratio_mean"].mean()
        ax.set_title(f"{METHOD_LABELS.get(method, method)}\n⟨ratio⟩={m:.4f}", fontsize=10)
        ax.set_xlabel("frac"); ax.set_ylabel("⟨reco/expected⟩")
        ax.legend(fontsize=7); ax.set_ylim(0.9, 1.1)

    savefig(fig, os.path.join(output_dir, "s4_spectra_recovery"), "S4: Spectra (summary)")


# ===================================================================
# Summary report
# ===================================================================

def write_summary(obs, methods, output_dir, input_file):
    n_iter = obs['iteration'].nunique()
    d_v5 = obs[obs["method"] == "smooth_v5"] if "smooth_v5" in methods else obs[obs["method"] == methods[0]]
    d_leg = obs[obs["method"] == "smooth"] if "smooth" in methods else d_v5

    report(f"\n{'='*70}")
    report("SCAN VALIDATION SUMMARY REPORT")
    report(f"{'='*70}")

    report(f"""
EXECUTIVE SUMMARY
  Scan: {n_iter} iterations, N in [{obs['N'].min()},{obs['N'].max()}], frac in [{obs['frac'].min():.2f},{obs['frac'].max():.2f}], dx in [{obs['dx'].min():.2f},{obs['dx'].max():.2f}]
  Reconstruction: Scw/N = {d_v5['sum_cw_over_N'].mean():.4f}+/-{d_v5['sum_cw_over_N'].std():.4f} (v5), {d_leg['sum_cw_over_N'].mean():.4f}+/-{d_leg['sum_cw_over_N'].std():.4f} (legacy) -- unbiased
  PDF bias (core): RMS = {d_v5['rms_bias_core'].mean():.4f} (v5) vs {d_leg['rms_bias_core'].mean():.4f} (legacy) -- v5 {d_leg['rms_bias_core'].mean()/d_v5['rms_bias_core'].mean():.1f}x better
  Spectra recovery: {d_v5['spectra_ratio_mean'].mean():.4f}+/-{d_v5['spectra_ratio_mean'].std():.4f} (v5) -- consistent with 1.0
  Weight efficiency: N_eff/N_sampled = {d_v5['neff_over_nsamp'].mean():.3f}+/-{d_v5['neff_over_nsamp'].std():.3f}
""")

    report(f"Input: {input_file}")
    report(f"Iterations: {n_iter}")
    report(f"Methods: {', '.join(methods)}")
    report(f"\nScan parameter ranges:")
    report(f"  N:    [{obs['N'].min()}, {obs['N'].max()}] (log-uniform)")
    report(f"  frac: [{obs['frac'].min():.4f}, {obs['frac'].max():.4f}] (uniform)")
    report(f"  dx:   [{obs['dx'].min():.4f}, {obs['dx'].max():.4f}] (uniform)")

    report(f"\n{'='*70}")
    report("PER-METHOD SUMMARY")
    report(f"{'='*70}")

    rows = []
    for method in methods:
        d = obs[obs["method"] == method]
        rows.append([
            method, len(d),
            f"{d['sum_cw_over_N'].mean():.4f}+/-{d['sum_cw_over_N'].std():.4f}",
            f"{d['frac_sampled'].mean():.4f}+/-{d['frac_sampled'].std():.4f}",
            f"{d['rms_bias_core'].mean():.4f}+/-{d['rms_bias_core'].std():.4f}",
            f"{d['spectra_ratio_mean'].mean():.4f}+/-{d['spectra_ratio_mean'].std():.4f}",
            f"{d['neff_over_nsamp'].mean():.3f}+/-{d['neff_over_nsamp'].std():.3f}",
        ])
    report_table(["Method", "N_iter", "Scw/N", "frac_samp",
                  "RMS_bias", "spectra", "Neff/Ns"], rows)

    # Figure descriptions
    report(f"\n{'='*70}")
    report("FIGURE DESCRIPTIONS -- PRIMARY")
    report(f"{'='*70}")

    report("""
S1: Sampling Fraction Accuracy (s1_frac_accuracy.png)
  y = asinh(N_sampled/(N*frac) - 1), centered at 0
  x-axis: frac (5 quantile bins), color: 3 log(N) bins
  Dashed lines: expected width +/- 1/sqrt(N_median * frac)
  Expected: all curves flat at 0, width matches 1/sqrt(N*frac)
""")
    report("""
S2: Threshold Stability (s2_threshold_stability.png -- 2 rows x 3 cols)
  Rows = methods, Cols = Dx bins. Shared axes for comparison.
  Within each (N_bin x frac_bin) cell: local linear fit thr = a + b*frac + c/sqrt(N)
  y = sigma(thr_obs - thr_fit) per cell -- one point per cell
  x-axis: 1/sqrt(N*frac), color: 3 frac bins
  Fit: y = slope/sqrt(N*frac) + intercept (black dashed)
  Key test: smaller Dx -> smaller intercept (less PDF bias -> less threshold offset)
  Expected: linear scaling in 1/sqrt(N*frac), offset decreasing with Dx
""")
    report("""
S3: PDF Estimator Bias (s3_pdf_bias.png -- combined, 2 rows x 4 cols)
  Rows = methods (legacy, v5), Cols = Dx bins. Shared x/y range for comparison.
  y = RMS(pdf_emp/pdf_true - 1) per pdf_true bin
  x-axis: 1/sqrt(lambda), lambda = N * pdf_mid * dx (Poisson parameter)
  Color: 3 pdf_true bins -- should collapse if Poisson dominates
  Model: RMS = sqrt(a^2/lambda + b^2)
    a = Poisson coefficient (expect ~1)
    b = discretization floor (expect ~0 for v5, >0 for legacy)
  Black dashed: a/sqrt(lambda) fit. Red dotted: 2-param fit.
  Fit parameters in summary table.

S3b: PDF Estimator Bias vs Position (s3b_pdf_bias_vs_x.png)
  2 rows x 4 cols. Color = method.
  Row 0: all bins. Row 1: excluding edge bins (first/last PDF bin).
  Col 0: bias vs x. Col 1: RMS vs x. Col 2: bias vs dx. Col 3: RMS vs dx.
  Key test: excluding edges should remove tail bias, confirming edge extrapolation
  is the sole source of large RMS in tails for non-uniform binning.
  Table: mean RMS per |x| region × subset.
""")
    report("""
S4: Spectra Recovery Model
  Model formula for sigma(ratio) at position x:
    sigma_model = sqrt( <sigma_model_i^2> )  where average is over iterations in group
    sigma_model_i = 1/sqrt(N_i * min(f(x), thr_i) * dx_hist)
    Dense regime (f(x) > thr_i):  sigma_i = 1/sqrt(N_i * thr_i * dx_hist)
    Sparse regime (f(x) < thr_i): sigma_i = 1/sqrt(N_i * f(x) * dx_hist)
  Using RMS (not harmonic mean) because sigma_meas = std(ratio) = sqrt(<sigma_i^2>).

S4a: Spectra Recovery by frac (s4a_spectra_frac_{method}.png)
  Layout: 2 rows x 3 cols. Cols = frac bins, Color = N bins.
  Row 0: bias = mean(ratio - 1) vs x (markers, no lines)
  Row 1: sigma(ratio) vs x (markers + error bars + model dashed)
  Grouping by frac separates threshold values -> model matches better.

S4b: Spectra Recovery by Dx (s4b_spectra_dx_{method}.png)
  Layout: 2 rows x 3 cols. Cols = Dx bins, Color = N_sampled bins.
  Same rows as S4a (markers, error bars, model dashed).
  Grouping by N_sampled -> direct control variable for sigma.
  Report: <sigma_meas/sigma_model> per group in core (|x|<2sigma).

S4c: Sigma Scaling Validation (s4c_sigma_scaling_{method}.png)
  Layout: 3 rows x 3 cols. Cols = Dx bins.
  Row 0: sigma_meas / sigma_model vs x (markers, error bars, no lines)
    Color = N_sampled bins. Horizontal dotted = mean ratio per group.
  Row 1: scatter sigma_meas vs sigma_model (6 N_sampled bins for density)
    Color = 3 discrete |x| categories: core (|x|<1), shoulder (1<|x|<2), tail (|x|>2)
    Fit: y = a*x through origin, report slope and R^2.
  Row 2: pull histogram = (sigma_meas - sigma_model) / (sigma_meas / sqrt(2*(n-1)))
    sigma_err = sigma_meas / sqrt(2*(n-1)) — standard chi^2 error on sample std.
    sigma_model = sqrt(<sigma_model_i^2>) — RMS of per-iteration models (not harmonic mean).
    Color = same 3 |x| categories. Annotated with mean and RMS per category.
    Expected: mean~0, RMS~1.
  Tables: fit (slope, R^2), pull stats (mean, RMS) per (method, Dx, |x| bin).

  Common: x range [-3sigma, 3sigma], 30 bins, markers only (no connecting lines)
""")

    report(f"\n{'='*70}")
    report("OBSERVABLE DEFINITIONS")
    report(f"{'='*70}")
    report("""
Per-iteration observables (scan_observables.csv):
  N, frac, dx:      Scan parameters (input)
  frac_sampled:     N_sampled / N
  N_sampled, N_eff, neff_over_nsamp, sum_cw_over_N, threshold
  bias_core, rms_bias_core: mean and RMS of (pdf_emp/pdf_true - 1) in core
  rms_pdf_X_Y:      RMS bias in pdf_true bin [X,Y]
  lambda_X_Y:       lambda = N * pdf_mid * dx for that bin
  spectra_ratio_mean/std: reweighted/expected histogram ratio
  Weights: w_HT = max(1, pdf/thr), w_cal = 1/max(pdf, thr)
""")

    report_file = os.path.join(output_dir, "scan_summary.txt")
    with open(report_file, "w") as f:
        f.write("\n".join(_report_lines))
    print(f"\nSaved: {report_file}")


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--methods", type=str, default="smooth,smooth_v5")
    parser.add_argument("--x_range", type=float, nargs=2, default=None,
                       help="Analysis range [x_lo, x_hi]. Default: from params table or data.")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    os.makedirs(args.output, exist_ok=True)

    report("=" * 70)
    report("SCAN VALIDATION -- Parameter Scaling Laws")
    report("=" * 70)

    t0 = time.time()
    scan, params = load_scan(args.input)
    report(f"  Load time: {time.time() - t0:.1f}s")

    # Derive metadata: --x_range > params x_lo/x_hi > data min/max
    meta = derive_scan_metadata(scan, params=params, x_range=args.x_range)
    source = "CLI" if args.x_range else ("params" if params is not None and "x_lo" in params.columns else "data")
    report(f"  x range: [{meta['x_lo']:.2f}, {meta['x_hi']:.2f}] (from {source}), "
           f"core<{meta['b1']:.2f}, shoulder<{meta['b2']:.2f}")

    t1 = time.time()
    obs = compute_iteration_observables(scan, params, methods, meta=meta)
    report(f"Computed: {len(obs)} rows ({obs['iteration'].nunique()} iter x {len(methods)} methods)")
    report(f"  Compute time: {time.time() - t1:.1f}s")

    obs_file = os.path.join(args.output, "scan_observables.csv")
    obs.to_csv(obs_file, index=False, float_format="%.6f")
    report(f"Saved: {obs_file}")

    report(f"\nGenerating figures...")
    figure_s1(obs, methods, args.output)
    figure_s2(obs, methods, args.output, scan=scan)
    figure_s3(obs, methods, args.output)
    # Edge-excluded S3 if isEdge data available
    if scan is not None and "isEdge" in scan.columns:
        figure_s3(obs, methods, args.output,
                  col_suffix="_noedge", label_suffix=" (excluding edge bins)")
    figure_s3b(obs, methods, args.output, scan=scan, params=params, meta=meta)
    figure_s4(obs, methods, args.output, scan=scan, params=params, meta=meta)

    save_combined_pdf(args.output)
    write_summary(obs, methods, args.output, args.input)

    report(f"\nTotal time: {time.time() - t0:.1f}s")
    report("=" * 70)


if __name__ == "__main__":
    main()
